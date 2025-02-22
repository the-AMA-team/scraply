import { type Config } from "tailwindcss";
import { fontFamily } from "tailwindcss/defaultTheme";

const colorSafelist = ["blue", "zinc"];

export default {
  safelist: colorSafelist
    .map((color) => [`bg-${color}-600`, `ring-${color}-600`])
    .flat(),
  content: ["./src/**/*.tsx"],
  theme: {
    extend: {
      fontFamily: {
        sans: ["var(--font-geist-sans)", ...fontFamily.sans],
      },
    },
  },
  plugins: [],
} satisfies Config;
