# VLA Lens Frontend

The frontend is a Vite/React workbench for the local dashboard API. The detailed
architecture and contract live in [../docs/workbench-frontend.md](../docs/workbench-frontend.md).

Useful commands:

```bash
npm ci --prefix frontend
npm run lint --prefix frontend
npm run test --prefix frontend
npm run build --prefix frontend
```

For a complete local demo with a synthetic dataset, backend, and dev server:

```bash
scripts/run_vla_lens_demo.sh
```
