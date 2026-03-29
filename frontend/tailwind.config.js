/** @type {import('tailwindcss').Config} */
module.exports = {
  darkMode: 'class',
  content: ['./src/**/*.{js,ts,jsx,tsx,mdx}'],
  theme: {
    extend: {
      fontFamily: {
        sans:    ['var(--font-body)', 'system-ui', 'sans-serif'],
        display: ['var(--font-display)', 'system-ui', 'sans-serif'],
        mono:    ['var(--font-mono)', 'monospace'],
      },
      colors: {
        surface: {
          DEFAULT: '#eef2f7',   // blue-grey page background (matches reference)
          dark:    '#0f0f11',
        },
        panel: {
          DEFAULT: '#ffffff',
          dark:    '#18181b',
        },
        border: {
          DEFAULT: '#e2e8f0',   // slightly cooler border to match blue theme
          dark:    '#27272a',
        },
        accent: {
          DEFAULT: '#3b3fd8',   // indigo-blue (matches reference buttons)
          light:   '#6366f1',   // lighter indigo for dark mode
          muted:   '#e0e7ff',   // indigo tint for pill backgrounds
        },
        ink: {
          DEFAULT: '#1c1917',
          muted:   '#78716c',
          faint:   '#a8a29e',
        },
      },
      animation: {
        'fade-in':      'fadeIn 0.3s ease-out',
        'slide-up':     'slideUp 0.35s cubic-bezier(0.16, 1, 0.3, 1)',
        'pulse-dot':    'pulseDot 1.4s ease-in-out infinite',
        'shimmer':      'shimmer 1.8s ease-in-out infinite',
        'token-appear': 'tokenAppear 0.15s ease-out',
      },
      keyframes: {
        fadeIn: {
          from: { opacity: '0' },
          to:   { opacity: '1' },
        },
        slideUp: {
          from: { opacity: '0', transform: 'translateY(12px)' },
          to:   { opacity: '1', transform: 'translateY(0)' },
        },
        pulseDot: {
          '0%, 80%, 100%': { transform: 'scale(0.6)', opacity: '0.4' },
          '40%':           { transform: 'scale(1)',   opacity: '1' },
        },
        shimmer: {
          '0%':   { backgroundPosition: '-200% 0' },
          '100%': { backgroundPosition: '200% 0' },
        },
        tokenAppear: {
          from: { opacity: '0', transform: 'translateY(2px)' },
          to:   { opacity: '1', transform: 'translateY(0)' },
        },
      },
    },
  },
  safelist: [
    'bg-accent-DEFAULT/10', 'bg-accent-DEFAULT/20', 'bg-accent-DEFAULT/30',
    'border-accent-DEFAULT/20', 'border-accent-DEFAULT/40', 'border-accent-DEFAULT/60',
    'hover:border-accent-DEFAULT/40', 'hover:border-accent-DEFAULT/50',
    'dark:bg-accent-DEFAULT/20', 'dark:border-accent-light/30', 'dark:border-accent-light/40',
    'focus-within:border-accent-DEFAULT/60', 'dark:focus-within:border-accent-light/40',
    'hover:bg-accent-DEFAULT/90', 'hover:bg-accent-muted/30',
    'bg-accent-muted/30',
  ],
  plugins: [],
}