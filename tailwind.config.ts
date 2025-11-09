import type { Config } from 'tailwindcss'

const config: Config = {
  content: [
    './pages/**/*.{js,ts,jsx,tsx,mdx}',
    './components/**/*.{js,ts,jsx,tsx,mdx}',
    './app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        forest: {
          DEFAULT: '#2d5016',
          light: '#3d5a2c',
        },
        moss: {
          DEFAULT: '#5a7c3e',
          light: '#6a8c4e',
          dark: '#4a6b2e',
        },
        cream: {
          DEFAULT: '#f5f1e8',
          light: '#faf8f3',
          dark: '#e8e4d9',
        },
        bark: {
          DEFAULT: '#5a4a3a',
          light: '#8b7355',
          dark: '#3d3d3d',
        },
        amber: '#d4a574',
      },
      fontFamily: {
        serif: ['Georgia', 'serif'],
        sans: ['Inter', 'sans-serif'],
      },
    },
  },
  plugins: [],
}

export default config