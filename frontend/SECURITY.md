React2Shell (CVE-2025-55182) mitigation notes
===============================================

Summary
-------
This frontend uses Next.js with React Server Components (RSC). React2Shell is a critical remote code execution (RCE) vulnerability affecting RSC and react-server-dom packages when running vulnerable React versions (19.0, 19.1.0, 19.1.1, 19.2.0).

Action taken
------------
- Upgraded `react` and `react-dom` to `19.2.1` in `package.json` to apply security patches.
- Allowed Next.js to upgrade to a compatible patch release by changing `next` to ^15.2.5 in `package.json`; ensure Next.js and its vendored react-rsc dependencies are updated.

Recommended next steps
----------------------
1. Regenerate the lockfile and install the patched versions:

   - pnpm (preferred):
     ```bash
     cd frontend
     pnpm install
     pnpm up --latest
     pnpm install
     ```

   - npm:
     ```bash
     cd frontend
     npm install
     npm update
     ```

2. Verify package versions:

   ```bash
   cd frontend
   pnpm list react react-dom next
   # or
   npm ls react react-dom next
   ```

3. Redeploy the frontend and run smoke tests. Confirm the build and app run correctly.

4. Add web application firewall (WAF) and IDS rules to detect and block Flight/RSC payloads until fully patched and validated:
   - Block or log requests with header `RSC` or `Next-Action`.
   - Block or log requests where `Content-Type: text/x-component` or query param `_rsc` is present.

5. Add `npm audit`/`pnpm audit` steps to CI and add a security gating check to prevent pulling in versions of `react` in the vulnerable ranges.

Notes
-----
This file documents mitigation steps for the RSC vulnerability (React2Shell). If the deployment uses a CDN or other proxy, you can leverage WAF rules at that layer to temporarily mitigate risk.
