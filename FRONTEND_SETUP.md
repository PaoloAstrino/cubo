# Frontend Setup Guide

This document outlines the requirements and steps to set up and run the CUBO frontend application.

## Prerequisites

- **Node.js**: Version 20 or higher is required (compatible with Next.js 15 and React 19).
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
