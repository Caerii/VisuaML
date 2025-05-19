/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        primary: {
          50: '#e5faff',
          500: '#06b6d4',
          700: '#0e7490',
        },
        accent: {
          400: '#fbbf24',
        },
      },
      borderRadius: {
        xl: '1rem',
        '2xl': '1.5rem',
      },
    },
  },
  plugins: [],
};
