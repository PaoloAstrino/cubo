# Frontend Setup Guide

This document outlines the requirements and steps to set up and run the CUBO frontend application.

## Prerequisites

- **Node.js**: Version 20 or higher is required (compatible with Next.js 15 and React 19).
## Security Notes

This project uses Next.js App Router and React Server Components (RSC). A critical vulnerability (React2Shell, CVE-2025-55182) affects some React RSC and react-server-dom versions; ensure React and ReactDOM are upgraded to 19.2.1 or later. After updating dependencies, run `npm audit` (or `pnpm audit`) and update lockfiles before redeploying. Add WAF/IDS rules to block Flight/RSC payloads until your deployment has been validated and rebuilt.

- **npm**: Comes with Node.js.

## Environment Variables

The frontend application requires the following environment variables. You can set them in a `.env` file in the `frontend/` directory (copy from `frontend/.env.example`).

| Variable | Description | Default |
|----------|-------------|---------|
| `NEXT_PUBLIC_API_URL` | The URL of the backend API server. | `http://localhost:8000` |

## Running the Application

You can run the following commands from the **root** of the repository:

### Installation
First, install the frontend dependencies (using `--legacy-peer-deps` due to React 19):
```bash
npm run install:frontend
```

### Development
To start the development server:
```bash
npm run dev
```
This will start the Next.js development server, typically at `http://localhost:3000`.

### Build
To build the application for production:
```bash
npm run build
```

### Start Production Server
To start the production server after building:
```bash
npm run start
```

### Linting
To run the linter:
```bash
npm run lint
```
