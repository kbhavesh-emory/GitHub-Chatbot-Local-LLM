// frontend/vite.config.js
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  base: '/', // serve from root (prevents chunk 404 when behind hostnames)
  server: {
    host: '0.0.0.0',
    port: 5173,
    strictPort: true,
    allowedHosts: [
      'einstein.neurology.emory.edu',
      '170.140.138.40',
      '192.168.70.17',
      'localhost',
      '127.0.0.1',
      '10.110.78.73'
    ],
    proxy: {
      // All frontend calls to /api/* are proxied to FastAPI on :8000
      '/api': {
        target: 'http://einstein.neurology.emory.edu:8000',
        changeOrigin: true,
        rewrite: (p) => p.replace(/^\/api/, '')
      }
    },
    fs: { allow: ['..'] }
  },
  build: {
    outDir: 'dist',
    sourcemap: false,
    chunkSizeWarningLimit: 1000
  }
})
