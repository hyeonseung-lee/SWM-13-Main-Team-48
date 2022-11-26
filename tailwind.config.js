/** @type {import('tailwindcss').Config} */
const colors = require("tailwindcss/colors");

module.exports = {
  important: true,
  // Active dark mode on class basis
  darkMode: "class",
  i18n: {
    locales: ["en-US"],
    defaultLocale: "en-US",
  },

  content: [
    "./public/**/*.html",
    "./src/**/*.{html,js}",
    "./src/**/*.{js,jsx,ts,tsx,vue}",
    "./pages/**/*.tsx",
    "./components/**/*.tsx",
    "./templates/**/*.html",
    "./node_modules/flowbite/**/*.js",
  ],
  // These options are passed through directly to PurgeCSS

  theme: {
    extend: {
      colors: {
        primary: {
          light: "#E8E2FD", // For lighter primary color
          DEFAULT: "#5900D0", // Normal primary color
          dark: "#3A0682", // Used for hover, active, etc.
        },
      },
    },
  },
  variants: {
    extend: {
      backgroundColor: ["checked"],
      borderColor: ["checked"],
      inset: ["checked"],
      zIndex: ["hover", "active"],
    },
  },
  plugins: [
    require("tailwindcss"),
    require("@tailwindcss/aspect-ratio"),
    require("flowbite/plugin"),
    require("kutty"),
  ],
  future: {
    purgeLayersByDefault: true,
  },
};
