/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        // Light mode
        bg: '#F4F8F9',
        surface: '#FFFFFF',
        border: '#E3EEF1',
        ink: '#0E2E36',
        muted: '#5B7C84',
        subtle: '#8AA6AD',
        accent: '#0B6B7D',
        'accent-light': '#E6F3F5',
        ok: '#0D7A4E',
        'ok-bg': '#E3F5EC',
        warn: '#B4610F',
        'warn-bg': '#FFF1DE',
        bad: '#C8462A',
        'bad-bg': '#FCEAE7',
        blue: '#2F6696',
        'blue-bg': '#E6F0F8',
      },
      fontFamily: {
        sans: ['Mulish', 'sans-serif'],
        mono: ['"Space Mono"', 'monospace'],
        serif: ['"Source Serif 4"', 'Georgia', 'serif'],
        display: ['Fraunces', 'serif'],
      },
    },
  },
  plugins: [],
}
