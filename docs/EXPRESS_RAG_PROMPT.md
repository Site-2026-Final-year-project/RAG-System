# Prompt for `driver-garage-backend` (optional hardening)

Copy into Cursor on the Express repo if you want tighter integration with the RAG service.

---

**Prompt:**

Our Python RAG service (`RAG-System`) now syncs admin education PDFs from the same Postgres database. It reads Prisma table `"EducationContent"` where `category` is `MANUALS` and `pdf_url` is set, resolves files under `uploads/education-manuals/` via `EXPRESS_PROJECT_ROOT`, or downloads the public `pdf_url`.

Please review `driver-garage-backend` and:

1. **Confirm** `EducationContent` rows for `MANUALS` always store a full `pdf_url` (including after create/update) and that `express.static` serves `/uploads` as today (`server.ts`).

2. **Optional (recommended):** Add fields on `EducationContent`: `ragIndexedAt DateTime?` and `ragIndexError String?` — set by a small internal webhook or by the Python sync script via a future `POST /internal/rag-index-callback` (optional). For MVP, manual re-run of `scripts/sync_education_manuals.py` is enough.

3. **Optional:** After admin creates/updates a `MANUALS` item, call an internal HTTP endpoint on the RAG service or enqueue a job that runs `sync_education_manuals.py --content-id <id>` — only if we need near-real-time indexing.

4. **JWT:** Ensure driver JWT `sub` matches `Driver.id` so Flutter → Python RAG chat uses the same user id as Express.

5. **Do not** move PDF storage without updating `pdf_url` resolution in Python (`EXPRESS_PROJECT_ROOT` + path under `uploads/education-manuals/`).

---

Nothing is **required** on Express for the current sync script to work, as long as `DATABASE_URL` is shared, Prisma migrations have been applied to that database (`npx prisma migrate deploy`), the `vector` extension exists, and the RAG process can read PDFs (same machine + `EXPRESS_PROJECT_ROOT`, or reachable `http(s)` `pdf_url`).
