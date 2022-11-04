module.exports = {
  important: true,
  // Active dark mode on class basis
  darkMode: "class",
  i18n: {
    locales: ["en-US"],
    defaultLocale: "en-US",
  },
  purge: {
    content: [
      "./pages/**/*.tsx",
      "./components/**/*.tsx",
      "./templates/**/*.html",
      "./node_modules/flowbite/**/*.js",
    ],
    // These options are passed through directly to PurgeCSS
  },
  theme: {
    extend: {
      backgroundImage: (theme) => ({
        check: "url('/icons/check.svg')",
        landscape: "url('/images/landscape/2.jpg')",
      }),
      colors: {
        primary: {
          light: "#fefcbf", // For lighter primary color
          DEFAULT: "#b7791f", // Normal primary color
          dark: "#744210", // Used for hover, active, etc.
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
    require("@tailwindcss/aspect-ratio"),
    require("flowbite/plugin"),
    require("kutty"),
  ],
  future: {
    purgeLayersByDefault: true,
  },
};
