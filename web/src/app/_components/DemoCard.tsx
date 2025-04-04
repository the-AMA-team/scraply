import { motion } from "framer-motion";
import {
  RxCross2 as CrossIcon,
  RxArrowLeft as LeftIcon,
  RxArrowRight as RightIcon,
} from "react-icons/rx";

const NavigationButton: React.FC<{
  children: React.ReactNode;
  onClick: () => void;
}> = ({ children, onClick }) => {
  return (
    <button onClick={onClick} className="mx-1 rounded-lg bg-zinc-600 px-2 py-1">
      {children}
    </button>
  );
};

interface DemoCardProps {
  title: string;
  description: string;
  currIdx: number;
  maxIdx: number;
  prev: boolean;
  onPrev: () => void;
  next: boolean;
  onNext: () => void;
  x: number;
  y: number;
  component: React.RefObject<HTMLDivElement>;
  closeDemo: () => void;
}

const DemoCard: React.FC<DemoCardProps> = ({
  title,
  description,
  currIdx,
  maxIdx,
  prev,
  onPrev,
  next,
  onNext,
  x,
  y,
  component,
  closeDemo,
}) => {
  return (
    <motion.div
      initial={{
        opacity: 0,
        left: component.current?.getBoundingClientRect().x || 0 + x,
        top: component.current?.getBoundingClientRect().y || 0 + y,
      }}
      animate={{
        opacity: 1,
        left: component.current?.getBoundingClientRect().x || 0 + x,
        top: component.current?.getBoundingClientRect().y || 0 + y,
      }}
      exit={{ opacity: 0 }}
      transition={{ duration: 0.15, ease: "easeOut" }}
      className="absolute max-w-fit rounded-xl bg-zinc-700 px-3 py-2 shadow-2xl ring ring-zinc-600"
    >
      <div className="mb-2 flex justify-between">
        <div className="text-lg font-semibold">{title}</div>
        <NavigationButton onClick={closeDemo}>
          <CrossIcon />
        </NavigationButton>
      </div>
      <div className="max-w-52 text-sm">{description}</div>
      <div className="my-4 flex justify-center text-sm">
        {prev && (
          <NavigationButton onClick={onPrev}>
            <LeftIcon />
          </NavigationButton>
        )}
        <div className="mx-2 py-1">
          {currIdx + 1} of {maxIdx + 1}
        </div>
        {next ? (
          <NavigationButton onClick={onNext}>
            <RightIcon />
          </NavigationButton>
        ) : (
          <NavigationButton onClick={closeDemo}>
            <CrossIcon />
          </NavigationButton>
        )}
      </div>
    </motion.div>
  );
};

export default DemoCard;
