import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/query': 'http://localhost:8000',
      '/regime': 'http://localhost:8000',
      '/features': 'http://localhost:8000',
      '/similarity': 'http://localhost:8000',
      '/anomaly': 'http://localhost:8000',
      '/feedback': 'http://localhost:8000',
      '/models': 'http://localhost:8000',
      '/health': 'http://localhost:8000',
    },
  },
})
