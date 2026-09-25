import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// In development the frontend talks to the backend through this proxy, so the
// browser only ever sees same-origin /api requests — the same shape nginx
// serves in production, which keeps CORS out of the picture.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: process.env.VITE_BACKEND ?? 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
      '/health': { target: process.env.VITE_BACKEND ?? 'http://127.0.0.1:8000', changeOrigin: true },
      '/metrics': { target: process.env.VITE_BACKEND ?? 'http://127.0.0.1:8000', changeOrigin: true },
    },
  },
  build: {
    outDir: 'dist',
    chunkSizeWarningLimit: 900,
  },
})
