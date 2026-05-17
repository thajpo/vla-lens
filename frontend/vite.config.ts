import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const backendUrl = process.env.VLA_LENS_BACKEND_URL ?? "http://127.0.0.1:8765";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": backendUrl,
    },
  },
});
