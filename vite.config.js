import { defineConfig } from 'vite';
export default defineConfig({base:'./',server:{allowedHosts:['terminal.local']},build:{target:'es2022',outDir:'dist',chunkSizeWarningLimit:1600}});
