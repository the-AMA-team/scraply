import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split  # --> pip install scikit-learn
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from collections import Counter, defaultdict
import cv2  # --> pip install opencv-python
import time
import asyncio
import random
from params import DATALOADERS, LAYERS, ACTIVATIONS, LOSSES, OPTIMIZERS
import os
import copy
import tempfile
import numpy as np
from scipy.special import entr
import base64

# pau code recommendations:
# - use enums instead of 'conv1d', 'conv2d', etc
# - make code more readable through function names etc instead of commenting it like crazy
# - maybe make the image dictionary a class and have a serialize function to send in json or send in binary encoding method
# - more functions; someone should be able to get the jist of what is happening without having to get into the weeds of it too much
# - doc strings for functions
# - use a logging library instead of print statements (important for deployment)
# - watch videos on denesting code yippee


class ConditionalActivationSaver:
    def __init__(self, model):
        self.model = model

    def hook(self, module, input, output):
        if self.model.feature_save and not self.model.training:
            self.model.feature_maps[module] = output.detach().clone()


class DynamicModel(nn.Module):
    def __init__(self, layers):
        super().__init__()
        raw_layers = layers
        self.layer_list = []
        self.feature_save = False
        self.isConv = False  # set to true if the model has a convolutional layer

        # Initialize feature_maps dictionary to store convolutional layer outputs during inference
        self.feature_maps = {}

        for l in raw_layers:
            component = None

            layer_type = l["kind"]
            if layer_type in LAYERS.keys():
                layer_args = l["args"]
                if layer_type == "Linear":
                    i, o = layer_args
                    component = nn.Linear(i, o)

                elif layer_type in ["Conv1D", "Conv2D", "Conv3D"]:
                    dim = layer_args[0]
                    i, o, k_size, stride, padding = layer_args[1:]
                    component = LAYERS[layer_type](dim, i, o, k_size, stride, padding)
                    self.isConv = True

                elif layer_type in ["LSTM", "GRU", "RNN"]:
                    i, h_size, dropout = layer_args
                    component = LAYERS[layer_type](i, h_size, dropout)

                elif layer_type == "Dropout":
                    p = layer_args[0]
                    component = nn.Dropout(p)  # 1 arg

                elif layer_type == "Flatten":
                    start_dim = 1
                    end_dim = -1
                    component = nn.Flatten(start_dim, end_dim)
                    # forcing nn.Flatten(1,-1)

                elif layer_type in ["MaxPool1D", "MaxPool2D", "MaxPool3D"]:
                    dim = layer_args[0]
                    k_size, stride, padding = layer_args[1:]
                    component = LAYERS[layer_type](dim, k_size, stride, padding)

                elif layer_type in ["AvgPool1D", "AvgPool2D", "AvgPool3D"]:
                    dim = layer_args[0]
                    k_size, stride, padding = layer_args[1:]
                    component = LAYERS[layer_type](dim, k_size, stride, padding)

                if component is None:
                    print(f"Layer {layer_type} not recognized or not implemented.")

            elif layer_type in ACTIVATIONS.keys():
                # Copy so concurrent jobs do not share one Module instance
                component = copy.deepcopy(ACTIVATIONS[layer_type])

            else:
                print("Invalid layer type")
                break

            self.layer_list.append(component)

        self.layers = nn.ModuleList(self.layer_list)

        # register hooks for convolutional layers only
        saver = ConditionalActivationSaver(self)
        saver = ConditionalActivationSaver(self)
        self._register_conv_hooks(saver.hook)

    def _register_conv_hooks(self, hook_fn):
        """Register forward hooks on all convolutional layers"""
        for layer in self.layers:
            if isinstance(layer, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                layer.register_forward_hook(hook_fn)

    def get_feature_maps(self):
        """Return the current feature maps dictionary"""
        return self.feature_maps

    def clear_feature_maps(self):
        """Clear the feature maps dictionary"""
        self.feature_maps.clear()

    def forward(self, x):
        for l in self.layers:
            x = l(x)

        return x


class Train:
    def __init__(self, model, input, loss, optimizer, batch_size):
        self.model = model
        self.input = input
        self.num_classes = 0
        ds = DATALOADERS[input]

        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        print(f"Using {self.device} device")

        # MOVE MODEL TO DEVICE
        self.model = model.to(self.device)

        # preprocessing data here!!!
        if self.input == "pima":
            self.num_classes = 2
            X = ds["X"]
            y = ds["y"]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
            y_train_tensor = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)

            # Reshape for binary classification
            X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
            y_test_tensor = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)
            # creating dataset and dataloaders
            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
            self.train_loader = DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True
            )
            self.test_loader = DataLoader(
                test_dataset, batch_size=batch_size, shuffle=False
            )

        else:
            train_set = ds["train"]
            test_set = ds["test"]
            self.num_classes = (
                len(train_set.classes)
                if hasattr(train_set, "classes")
                else len(torch.unique(torch.tensor([label for _, label in train_set])))
            )

            self.train_loader = DataLoader(
                train_set, batch_size=batch_size, shuffle=True
            )
            self.test_loader = DataLoader(
                test_set, batch_size=batch_size, shuffle=False
            )

        self.loss_fn = copy.deepcopy(LOSSES[loss])
        self.optimizer = OPTIMIZERS[optimizer["kind"]](
            self.model.parameters(), optimizer["lr"]
        )
        self.final_loss = -1

    def train(self, n_epochs, batch_size, active_training=None):
        self.model.feature_save = False  # set the feature_save flag to false
        self.model.train()
        train_loss = 0
        correct = 0
        total = 0
        processed_batches = 0

        for batch, (X, y) in enumerate(self.train_loader):
            # Allow pause/stop mid-epoch (important for slower datasets like MNIST/CIFAR)
            if active_training is not None:
                # Stop requested
                if not active_training.get("is_training", True):
                    break
                # Pause requested
                while active_training.get("is_paused", False):
                    # Mark pause as confirmed once we actually enter the wait loop
                    if not active_training.get("pause_confirmed", False):
                        active_training["pause_confirmed"] = True
                    time.sleep(0.1)
                    if not active_training.get("is_training", True):
                        break
                # Clear confirmation once we're out of the pause loop
                if active_training.get("pause_confirmed", False) and not active_training.get(
                    "is_paused", False
                ):
                    active_training["pause_confirmed"] = False
                if not active_training.get("is_training", True):
                    break

            X, y = X.to(self.device), y.to(self.device)
            # Compute prediction error
            pred = self.model(X)
            loss = self.loss_fn(pred, y)
            # Backpropagation
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            train_loss += loss.item()
            processed_batches += 1

            if self.input == "pima":
                threshold = 0.5
                predicted = (pred > threshold).float()  # binary classification
            else:
                _, predicted = torch.max(pred, 1)  # multi-class classification

            correct += (predicted == y).sum().item()
            total += y.size(0)

        # avg over all batches
        if processed_batches == 0 or total == 0:
            return 0.0, 0.0

        avg_train_loss = train_loss / processed_batches
        avg_acc = 100 * correct / total
        return avg_train_loss, avg_acc

    def test(self, output_info=False, active_training=None):
        self.model.feature_save = False
        self.model.eval()  # set model to eval mode

        all_predictions, all_labels, all_indices = [], [], []
        test_loss, correct, total = 0, 0, 0
        processed_batches = 0

        with torch.no_grad():
            for idx, (X, y) in enumerate(self.test_loader):
                # Allow pause/stop mid-eval as well (so "Stop" works immediately)
                if active_training is not None:
                    if not active_training.get("is_training", True):
                        break
                    while active_training.get("is_paused", False):
                        if not active_training.get("pause_confirmed", False):
                            active_training["pause_confirmed"] = True
                        time.sleep(0.1)
                        if not active_training.get("is_training", True):
                            break
                    if active_training.get("pause_confirmed", False) and not active_training.get(
                        "is_paused", False
                    ):
                        active_training["pause_confirmed"] = False
                    if not active_training.get("is_training", True):
                        break

                X, y = X.to(self.device), y.to(self.device)
                pred = self.model(X)
                test_loss += self.loss_fn(pred, y).item()
                processed_batches += 1

                if self.input == "pima":
                    threshold = 0.5
                    predicted = (pred > threshold).type(
                        torch.float
                    )  # binary classification
                    correct += (predicted == y).sum().item()

                else:
                    _, predicted = torch.max(pred, 1)  # multi-class classification
                    correct += (predicted == y).sum().item()

                total += y.size(0)
                all_predictions.extend(predicted.cpu().numpy().tolist())
                all_labels.extend(y.cpu().numpy().tolist())
                all_indices.extend([idx * y.size(0) + i for i in range(y.size(0))])

        # calculate metrics for all datasets
        per_class_metrics = self.calculate_per_class_metrics(
            all_labels, all_predictions
        )
        confusion_matrix_data = self.calculate_confusion_matrix(
            all_labels, all_predictions
        )
        overall_metrics = self.calculate_overall_metrics(all_labels, all_predictions)

        if output_info and self.input != "pima":
            # calculate per-class accuracy
            class_correct = defaultdict(int)
            class_total = defaultdict(int)
            class_predictions = defaultdict(list)
            # class_predictions is a dictionary of lists. each list contains (idx, true_label, pred_label)
            # ex: class_predictions[class_label] = [(idx, true_label, pred_label), (idx, true_label, pred_label), (idx, true_label, pred_label)]

            for idx, (true_label, pred_label) in enumerate(
                zip(all_labels, all_predictions)
            ):
                class_total[true_label] += 1
                class_predictions[true_label].append((idx, true_label, pred_label))

                if true_label == pred_label:
                    class_correct[true_label] += 1

            # get 3 lowest class accuracies
            accuracies = per_class_metrics["accuracy"]
            lowest_classes = sorted(
                range(len(accuracies)), key=lambda i: accuracies[i]
            )[:3]
            lowest_accuracy_classes_info = {
                c: class_predictions[c] for c in lowest_classes
            }
            print(f"\n3 lowest accuracy classes: {lowest_classes}")

            print("\nGetting random predictions per class...")
            random_samples = self.get_random_predictions_per_class(
                class_predictions, num_samples=3
            )

            print("Getting misclassified samples for lowest accuracy classes...")
            misclassified_samples = self.get_misclassified_samples(
                lowest_accuracy_classes_info, num_samples=3
            )

        # returning test loss here :)
        if processed_batches == 0 or total == 0:
            avg_test_loss = 0.0
            avg_acc = 0.0
        else:
            avg_test_loss = test_loss / processed_batches
            avg_acc = 100 * correct / total

        if output_info and self.input != "pima":
            return (
                avg_test_loss,
                avg_acc,
                random_samples,
                misclassified_samples,
                per_class_metrics,
                confusion_matrix_data,
                overall_metrics,
            )
        else:
            return (
                avg_test_loss,
                avg_acc,
                per_class_metrics,
                confusion_matrix_data,
                overall_metrics,
            )  # ORIGINAL OUTPUT

    def get_random_predictions_per_class(self, class_predictions, num_samples=3):
        random_samples = {}

        for class_label in range(10):
            if class_label in class_predictions:
                samples = class_predictions[class_label]
                if len(samples) >= num_samples:
                    random_samples[class_label] = random.sample(samples, num_samples)
                else:
                    random_samples[class_label] = samples

        return random_samples

    def get_misclassified_samples(self, class_predictions, num_samples=3):
        misclassified_samples = {}

        for class_label in range(self.num_classes):
            if class_label in class_predictions:
                # get only misclassified samples
                misclassified = [
                    sample
                    for sample in class_predictions[class_label]
                    if sample[1] != sample[2]
                ]

                if len(misclassified) >= num_samples:
                    misclassified_samples[class_label] = random.sample(
                        misclassified, num_samples
                    )
                else:
                    misclassified_samples[class_label] = misclassified

        return misclassified_samples

    def calculate_per_class_metrics(self, all_labels, all_predictions):
        # metrics: recall, precision, F1 score, and accuracy

        if len(all_labels) == 0 or len(all_predictions) == 0:
            return {"precision": [], "recall": [], "f1_score": [], "accuracy": []}

        # convert to numpy arrays
        labels = np.array(all_labels)
        predictions = np.array(all_predictions)

        precision, recall, f1, support = precision_recall_fscore_support(
            labels, predictions, average=None, zero_division="warn"
        )

        # Calculate per-class accuracy
        accuracy = []
        for class_label in range(self.num_classes):
            class_mask = labels == class_label
            if np.sum(class_mask) > 0:
                acc = np.sum((predictions == labels) & class_mask) / np.sum(class_mask)
            else:
                acc = 0.0
            accuracy.append(float(acc))

        # convert to lists
        precision = [
            float(x)
            for x in (precision.tolist() if hasattr(precision, "tolist") else precision)
        ]
        recall = [
            float(x) for x in (recall.tolist() if hasattr(recall, "tolist") else recall)
        ]
        f1 = [float(x) for x in (f1.tolist() if hasattr(f1, "tolist") else f1)]

        per_class_metrics = {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "accuracy": accuracy,
        }
        return per_class_metrics

    def calculate_confusion_matrix(self, all_labels, all_predictions):
        """Calculate confusion matrix"""
        if len(all_labels) == 0 or len(all_predictions) == 0:
            return []

        # convert to numpy arrays
        labels = np.array(all_labels)
        predictions = np.array(all_predictions)

        cm = confusion_matrix(labels, predictions)

        # Convert to list format for JSON serialization
        return cm.tolist()

    def calculate_overall_metrics(self, all_labels, all_predictions):
        """Calculate overall precision, recall, and F1 score for the entire test dataset"""
        if len(all_labels) == 0 or len(all_predictions) == 0:
            return {}

        # convert to numpy arrays
        labels = np.array(all_labels)
        predictions = np.array(all_predictions)

        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average="weighted", zero_division="warn"
        )
        overall_accuracy = np.sum(predictions == labels) / len(labels)

        overall_metrics = {
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "accuracy": float(overall_accuracy),
        }

        return overall_metrics

    def process_image_samples(self, samples_dict, base_dir, dev_testing=False):
        """Process, save, and return images and peek maps for a given set of samples.
        Args:
            samples_dict (dict): {class_label: [(idx, true_label, pred_label), ...]}
            base_dir (str): Directory to save images and peek maps
            dev_testing (bool): If True, actually save images to disk. Default False.
        Returns:
            dict: {class_label: [ {original, peek_maps, true_label, pred_label, idx}, ... ] }
            ** class_label is the true label of the image
        """
        if self.input == "pima":
            return {}

        if dev_testing:
            os.makedirs(base_dir, exist_ok=True)
            print(f"Processing samples in directory: {base_dir}")

        result = {}

        for class_label, samples in samples_dict.items():
            class_dir = os.path.join(base_dir, f"class_{class_label}")
            if dev_testing:
                os.makedirs(class_dir, exist_ok=True)
            result[class_label] = []

            for idx, true_label, pred_label in samples:
                image, _ = self.test_loader.dataset[idx]
                image = image.unsqueeze(0).to(self.device)

                # Get image dimensions
                if len(image.shape) == 4:
                    _, c, h, w = image.shape
                elif len(image.shape) == 3:
                    c, h, w = image.shape
                else:
                    h, w = image.shape
                    c = 1

                self.model.feature_save = True
                with torch.no_grad():
                    output = self.model(image)

                    # Save original image to disk (handle grayscale and RGB)
                    image_np = image[0].cpu().numpy()
                    if image_np.shape[0] == 1:
                        # Grayscale: (1, H, W) -> (H, W)
                        image_np = image_np[0]
                        image_np = (
                            (image_np - image_np.min())
                            * (255.0 / (image_np.max() - image_np.min()))
                        ).astype(np.uint8)
                        image_np_resized = cv2.resize(
                            image_np, (400, 400), interpolation=cv2.INTER_NEAREST
                        )
                        if dev_testing:
                            cv2.imwrite(
                                os.path.join(class_dir, f"image_{idx}_original.png"),
                                image_np_resized,
                            )
                    else:
                        # RGB: (3, H, W) -> (H, W, 3)
                        image_np = np.transpose(image_np, (1, 2, 0))
                        image_np = (
                            (image_np - image_np.min())
                            * (255.0 / (image_np.max() - image_np.min()))
                        ).astype(np.uint8)
                        image_np_resized = cv2.resize(
                            image_np, (400, 400), interpolation=cv2.INTER_NEAREST
                        )
                        if dev_testing:
                            cv2.imwrite(
                                os.path.join(class_dir, f"image_{idx}_original.png"),
                                cv2.cvtColor(image_np_resized, cv2.COLOR_RGB2BGR),
                            )

                    # Encode original image as base64
                    img_b64 = self.image_to_base64_png(image_np_resized)
                    image_data = {
                        "original": img_b64,
                        "peek_maps": [],
                        "true_label": int(true_label),
                        "pred_label": int(pred_label),
                        "idx": idx,
                    }

                    if self.model.isConv:
                        feature_maps = self.model.feature_maps

                        for layer, fmap in feature_maps.items():
                            fmap = fmap.cpu().numpy()
                            fmap = fmap[0]
                            fmap = np.moveaxis(fmap, 0, -1)
                            peek_map = self.compute_PEEK(fmap, h, w)
                            peek_map_norm = (peek_map - peek_map.min()) / (
                                peek_map.max() - peek_map.min()
                            )
                            peek_map_norm_resized = cv2.resize(
                                peek_map_norm,
                                (400, 400),
                                interpolation=cv2.INTER_NEAREST,
                            )
                            heatmap = cv2.applyColorMap(
                                (peek_map_norm_resized * 255).astype(np.uint8),
                                cv2.COLORMAP_JET,
                            )
                            if c == 1:
                                overlay = cv2.addWeighted(
                                    cv2.cvtColor(image_np_resized, cv2.COLOR_GRAY2BGR),
                                    0.3,
                                    heatmap,
                                    0.7,
                                    0,
                                )
                            else:
                                if image_np_resized.shape[2] == 3:
                                    base_img = image_np_resized
                                else:
                                    base_img = cv2.cvtColor(
                                        image_np_resized, cv2.COLOR_GRAY2BGR
                                    )
                                overlay = cv2.addWeighted(
                                    base_img, 0.3, heatmap, 0.7, 0
                                )

                            # Save peek map overlay to disk
                            if dev_testing:
                                cv2.imwrite(
                                    os.path.join(class_dir, f"image_{idx}_{layer}.png"),
                                    overlay,
                                )
                                print(
                                    f"Saved original image and peek map for class {true_label}, sample {idx}, predicted {pred_label}"
                                )

                            # Encode peek map overlay as base64
                            _, buffer = cv2.imencode(".png", overlay)
                            peek_b64 = base64.b64encode(buffer.tobytes()).decode(
                                "utf-8"
                            )
                            image_data["peek_maps"].append(
                                {"layer": str(layer), "image": peek_b64}
                            )

                    else:
                        if dev_testing:
                            print(
                                f"Saved original image for class {true_label}, sample {idx} (no convolutional layers)"
                            )

                self.model.feature_save = False
                self.model.clear_feature_maps()
                result[class_label].append(image_data)

        return result

    async def train_test_log_stream_async(
        self,
        n_epochs,
        batch_size,
        socketio=None,
        active_training=None,
        dev_testing=False,
        room=None,
    ):
        """Async version of train_test_log_stream for use with ASGI servers"""
        # intended for default to support streaming. however, socketio object and active_training must be passed in. they optional for dev_testing
        train_losses = []
        train_accs = []
        test_losses = []
        test_accs = []
        per_class_metrics = {}
        confusion_matrix_data = []
        overall_metrics = {}

        RANDOM_SAMPLES_ENCODED = {}
        MISCLASSIFIED_SAMPLES_ENCODED = {}
        job_id = (active_training or {}).get("job_id") or "default"
        sample_root = os.path.join(tempfile.gettempdir(), "scraply_jobs", str(job_id))

        async def _emit(event, data):
            if dev_testing or socketio is None:
                print(event, data)
                return
            # Never broadcast: a missing room would send this job's events to every client
            if not room:
                print(f"Skipping {event}: no job room (refusing global broadcast)")
                return
            payload = dict(data) if isinstance(data, dict) else {"data": data}
            if job_id != "default":
                payload["job_id"] = job_id
            await socketio.emit(event, payload, room=room)

        await _emit(
            "training_started",
            {
                "total_epochs": n_epochs,
                "dataset": self.input,
            },
        )

        async def _pause_state_notifier():
            """Emit pause/resume only when loop is actually paused/resumed."""
            # Initialize to current state to avoid emitting 'resumed' at start
            last = bool(active_training.get("pause_confirmed", False)) if active_training else False
            while active_training and active_training.get("is_training", False):
                now = bool(active_training.get("pause_confirmed", False))
                if now != last:
                    if now:
                        print("⏸️  Training paused (confirmed)")
                        await _emit(
                            "training_paused",
                            {"message": "Training is paused"},
                        )
                    else:
                        print("▶️  Training Resumed (confirmed)")
                        await _emit(
                            "training_resumed",
                            {"message": "Training is running"},
                        )
                    last = now
                await asyncio.sleep(0.1)

        pause_notifier_task = None
        if not dev_testing and socketio is not None and active_training is not None:
            pause_notifier_task = asyncio.create_task(_pause_state_notifier())

        for t in range(n_epochs):
            # Check for pause before starting epoch
            if dev_testing == False:
                pause_printed = False
                while active_training and active_training.get("is_paused", False):
                    
                    if not pause_printed:
                        print("⏸️  Training is paused - waiting for resume...")
                        # Confirm pause when we hit this wait loop (epoch boundary pause)
                        if active_training is not None:
                            active_training["pause_confirmed"] = True
                        pause_printed = True

                    await asyncio.sleep(0.1)  # Sleep briefly to avoid busy waiting
                    if not active_training.get("is_training", False):
                        # Training was stopped while paused
                        print("🛑 Training was stopped while paused")
                        await _emit(
                            "training_stopped", {"message": "Training stopped"}
                        )
                        if pause_notifier_task:
                            pause_notifier_task.cancel()
                        return
                # If we exited pause loop, clear confirmation
                if active_training is not None and active_training.get("pause_confirmed", False):
                    active_training["pause_confirmed"] = False
                # Check if training was stopped
                if not active_training or not active_training.get("is_training", False):
                    print("🛑 Training stopped before epoch completion")
                    await _emit(
                        "training_stopped", {"message": "Training stopped"}
                    )
                    if pause_notifier_task:
                        pause_notifier_task.cancel()
                    return

            print(f"Epoch {t + 1}/{n_epochs}...")
            await _emit(
                "epoch_started", {"epoch": t + 1, "total_epochs": n_epochs}
            )
            # emit is method to send events and data to clients via websocket
            avg_train_loss, train_avg_acc = await asyncio.to_thread(
                self.train, n_epochs, batch_size, active_training
            )

            # If stop was requested during training, exit ASAP
            if active_training and not active_training.get("is_training", False):
                await _emit("training_stopped", {"message": "Training stopped"})
                if pause_notifier_task:
                    pause_notifier_task.cancel()
                return

            print(
                f"Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_avg_acc:.2f}%\n"
            )

            if t != n_epochs - 1 or self.input == "pima":
                test_result = await asyncio.to_thread(
                    self.test, False, active_training
                )
                (
                    avg_test_loss,
                    test_avg_acc,
                    per_class_metrics,
                    confusion_matrix_data,
                    overall_metrics,
                ) = test_result
            else:
                test_result = await asyncio.to_thread(
                    self.test, True, active_training
                )
                if len(test_result) == 7:  # Non-pima dataset with output_info=True
                    (
                        avg_test_loss,
                        test_avg_acc,
                        random_samples,
                        misclassified_samples,
                        per_class_metrics,
                        confusion_matrix_data,
                        overall_metrics,
                    ) = test_result

                    # Process samples if available
                    if self.input != "pima":
                        print("----------processing random samples-----------")
                        RANDOM_SAMPLES_ENCODED = self.process_image_samples(
                            random_samples,
                            sample_root,
                            dev_testing=dev_testing,
                        )
                        print("----------processing misclassified samples-----------")
                        MISCLASSIFIED_SAMPLES_ENCODED = self.process_image_samples(
                            misclassified_samples,
                            os.path.join(sample_root, "lowest_accuracy_classes"),
                            dev_testing=dev_testing,
                        )
                    else:  # Fallback for 5-value return
                        (
                            avg_test_loss,
                            test_avg_acc,
                            per_class_metrics,
                            confusion_matrix_data,
                            overall_metrics,
                        ) = test_result

            # If stop was requested during eval, exit ASAP
            if active_training and not active_training.get("is_training", False):
                await _emit("training_stopped", {"message": "Training stopped"})
                if pause_notifier_task:
                    pause_notifier_task.cancel()
                return

            print(
                f"Test Loss: {avg_test_loss:.4f}, Test Accuracy: {test_avg_acc:.2f}%\n"
            )

            train_losses.append(avg_train_loss)
            train_accs.append(train_avg_acc)
            test_losses.append(avg_test_loss)
            test_accs.append(test_avg_acc)

            # Prepare progress data
            progress_data = {
                "epoch": t + 1,
                "total_epochs": n_epochs,
                "progress": ((t + 1) / n_epochs) * 100,
                "train_loss": avg_train_loss,
                "train_accuracy": train_avg_acc,
                "test_loss": avg_test_loss,
                "test_accuracy": test_avg_acc,
                "train_losses": [{"x": i, "y": v} for i, v in enumerate(train_losses)],
                "test_losses": [{"x": i, "y": v} for i, v in enumerate(test_losses)],
            }

            # Update active training state
            if active_training is not None:
                active_training["current_progress"] = progress_data

            # Emit epoch progress
            await _emit("epoch_completed", progress_data)

        # Calculate final averages
        avg_train_acc = sum(train_accs) / len(train_accs)
        avg_test_acc = sum(test_accs) / len(test_accs)
        avg_train_loss = sum(train_losses) / len(train_losses)
        avg_test_loss = sum(test_losses) / len(test_losses)

        print("Done!")

        if pause_notifier_task:
            pause_notifier_task.cancel()

        # Format losses for final result
        train_losses = [{"x": i, "y": v} for i, v in enumerate(train_losses)]
        test_losses = [{"x": i, "y": v} for i, v in enumerate(test_losses)]

        ORIGINAL_OUTPUT = {
            "train_losses": train_losses,
            "test_losses": test_losses,
            "avg_train_loss": avg_train_loss,
            "avg_test_loss": avg_test_loss,
            "avg_train_acc": avg_train_acc,
            "avg_test_acc": avg_test_acc,
        }

        RESULTS = {
            "training": ORIGINAL_OUTPUT,
            "outputs_class": per_class_metrics,
            "outputs_overall": overall_metrics,
            "confusion_matrix": confusion_matrix_data,
            "random_samples": RANDOM_SAMPLES_ENCODED,
            "top_misclassified": MISCLASSIFIED_SAMPLES_ENCODED,
        }

        # Emit training completion
        await _emit(
            "training_completed",
            {"final_results": RESULTS, "message": "Training completed successfully!"},
        )

        return RESULTS

    def compute_PEEK(self, feature_maps, h, w):
        """Compute PEEK map from feature maps"""
        positivized_maps = feature_maps + np.abs(np.min(feature_maps))
        entropy_map = -np.sum(entr(positivized_maps), axis=-1)
        peek_map = cv2.resize(entropy_map, (w, h))
        return peek_map

    def image_to_base64_png(self, image_np):
        """Encode a numpy image as base64 PNG string."""
        _, buffer = cv2.imencode(".png", image_np)
        return base64.b64encode(buffer.tobytes()).decode("utf-8")


if __name__ == "__main__":
    # example data
    dev_testing = True  # Set to True to enable image saving for development testing
    print("pima linear layer test")
    data = {
        "input": "pima",
        "layers": [
            {"kind": "Linear", "args": (8, 12)},
            {"kind": "ReLU"},
            {"kind": "Linear", "args": (12, 8)},
            {"kind": "ReLU"},
            {"kind": "Linear", "args": (8, 1)},
            {"kind": "Sigmoid"},
        ],
        "loss": "BCE",
        "optimizer": {"kind": "Adam", "lr": 0.001},
        "epoch": 3,
        "batch_size": 10,
    }

    inp = data["input"]
    layers = data["layers"]
    loss = data["loss"]
    optimizer = data["optimizer"]
    n_epochs = data["epoch"]
    batch_size = data["batch_size"]

    RESULTS = {}

    model = DynamicModel(layers)

    t = Train(
        model=model,
        input=inp,
        loss=loss,
        optimizer=optimizer,
        batch_size=batch_size,
    )

    # Pass dev_testing to process_image_samples via train_test_log if needed
    import asyncio

    RESULTS = asyncio.run(
        t.train_test_log_stream_async(
            n_epochs, batch_size, socketio=None, active_training=None, dev_testing=True
        )
    )

    print("Original Results:", RESULTS["training"])
    print("Per-class Metrics:", RESULTS["outputs_class"])
    print("Overall Metrics:", RESULTS["outputs_overall"])
    print("Confusion Matrix:", RESULTS["confusion_matrix"])
    print("Random samples: ", RESULTS["random_samples"])
    print("Top Misclassified: ", RESULTS["top_misclassified"])

    # print("mnist cnn test")

    # data = {
    #     "input": "MNIST",
    #     "layers": [
    #         {
    #             "kind": "Conv2D",
    #             "args": (2, 1, 16, 3, 1, 0),
    #         },  # dim, input, output, kernel size, stride, padding.
    #         # dim is a fake arg we made up so we just ignore it in the actual api. same for maxpool layers.
    #         {"kind": "ReLU"},
    #         {"kind": "MaxPool2D", "args": (2, 2, 2, 0)},
    #         {"kind": "Conv2D", "args": (2, 16, 32, 3, 1, 0)},
    #         {"kind": "ReLU"},
    #         {"kind": "MaxPool2D", "args": (2, 2, 2, 0)},
    #         {"kind": "Flatten", "args": [1, -1]},
    #         {"kind": "Linear", "args": (800, 128)},  # supposed to be 32 * 7 * 7
    #         {"kind": "ReLU"},
    #         {"kind": "Linear", "args": (128, 10)},
    #     ],
    #     "loss": "CrossEntropy",
    #     "optimizer": {"kind": "Adam", "lr": 0.001},
    #     "epoch": 2,
    #     "batch_size": 64,
    # }

    # inp = data["input"]
    # layers = data["layers"]
    # loss = data["loss"]
    # optimizer = data["optimizer"]
    # n_epochs = data["epoch"]
    # batch_size = data["batch_size"]

    # model = DynamicModel(layers)

    # t = Train(
    #     model=model,
    #     input=inp,
    #     loss=loss,
    #     optimizer=optimizer,
    #     batch_size=batch_size,
    # )

    # Pass dev_testing to process_image_samples via train_test_log if needed
    # RESULTS = t.train_test_log_stream(n_epochs, batch_size, socketio=None, active_training=None, dev_testing=True)

    # print("Original Results:", RESULTS["training"])
    # print("Per-class Metrics:", RESULTS["outputs_class"])
    # print("Overall Metrics:", RESULTS["outputs_overall"])
    # print("Confusion Matrix:", RESULTS["confusion_matrix"])
    # # print('Random samples: ', RESULTS['random_samples'])
    # print('Top Misclassified: ', RESULTS['top_misclassified'])

    print("cifar10 cnn test")

    data = {
        "input": "CIFAR10",
        "layers": [
            {
                "kind": "Conv2D",
                "args": (2, 3, 16, 3, 1, 1),
            },  # For CIFAR10: input channels=3 (RGB), output=16, kernel=3x3, stride=1, padding=1
            {"kind": "ReLU"},
            {
                "kind": "MaxPool2D",
                "args": (2, 2, 2, 0),
            },  # kernel=2, stride=2, padding=0
            {
                "kind": "Conv2D",
                "args": (2, 16, 32, 3, 1, 1),
            },  # input=16, output=32, kernel=3x3, stride=1, padding=1
            {"kind": "ReLU"},
            {"kind": "MaxPool2D", "args": (2, 2, 2, 0)},
            {"kind": "Flatten", "args": [1, -1]},
            {
                "kind": "Linear",
                "args": (8 * 8 * 32, 128),
            },  # 32 channels, 8x8 after pooling
            {"kind": "ReLU"},
            {"kind": "Linear", "args": (128, 10)},  # 10 classes for CIFAR10
        ],
        "loss": "CrossEntropy",
        "optimizer": {"kind": "Adam", "lr": 0.001},
        "epoch": 2,
        "batch_size": 64,
    }

    inp = data["input"]
    layers = data["layers"]
    loss = data["loss"]
    optimizer = data["optimizer"]
    n_epochs = data["epoch"]
    batch_size = data["batch_size"]

    model = DynamicModel(layers)

    t = Train(
        model=model,
        input=inp,
        loss=loss,
        optimizer=optimizer,
        batch_size=batch_size,
    )

    # Pass dev_testing to process_image_samples via train_test_log if needed
    RRESULTS = asyncio.run(
        t.train_test_log_stream_async(
            n_epochs, batch_size, socketio=None, active_training=None, dev_testing=True
        )
    )

    print("Original Results:", RESULTS["training"])
    print("Per-class Metrics:", RESULTS["outputs_class"])
    print("Overall Metrics:", RESULTS["outputs_overall"])
    print("Confusion Matrix:", RESULTS["confusion_matrix"])
    # print('Random samples: ', RESULTS['random_samples'])
    # print('Top Misclassified: ', RESULTS['top_misclassified'])
