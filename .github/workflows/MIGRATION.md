# CI/CD Workflow Migration Guide

## ✅ What Was Created

### 1. `.github/workflows/pr-checks.yml` (NEW)
**Purpose**: Fast feedback for Pull Requests
**Runtime**: ~3-5 minutes
**Jobs** (3 parallel):
- `lint` - Ruff, Black, isort, Bandit
- `unit-tests` - Fast unit tests only
- `import-check` - Smoke test core imports

**Triggers**: On every PR to main/master

---

### 2. `.github/workflows/main-validation.yml` (NEW)
**Purpose**: Comprehensive validation for main branch
**Runtime**: ~15-30 minutes
**Jobs** (6 parallel):
- `tests-ubuntu` - Full tests with coverage (Ubuntu + FAISS)
- `tests-windows` - Full tests (Windows, no FAISS)
- `integration-tests` - Integration + E2E + Playwright
- `deduplication-tests` - Dedup-specific tests
- `performance-benchmarks` - Performance tests
- `security-scan` - Full Bandit + Safety scan

**Triggers**: On push to main/master

---

## 🗑️ What to Delete (Old Workflows)

### Files to Remove:
1. `.github/workflows/ci-cd.yml` (957 lines → replaced)
2. `.github/workflows/dedup_tests.yml` (29 lines → merged into main-validation)
3. `.github/workflows/e2e.yml` (116 lines → merged into main-validation)

### How to Delete:
```bash
# Backup old workflows first (optional)
mkdir .github/workflows/archive
mv .github/workflows/ci-cd.yml .github/workflows/archive/
mv .github/workflows/dedup_tests.yml .github/workflows/archive/
mv .github/workflows/e2e.yml .github/workflows/archive/

# Or delete directly
rm .github/workflows/ci-cd.yml
rm .github/workflows/dedup_tests.yml
rm .github/workflows/e2e.yml
```

---

## 📊 Before vs After Comparison

### Before (3 workflows, 1102 total lines)
| File | Lines | Jobs | Runtime | Runs On |
|------|-------|------|---------|---------|
| `ci-cd.yml` | 957 | 13 sequential | ~60+ min | Every PR + Push |
| `dedup_tests.yml` | 29 | 1 | ~5 min | Dedup changes only |
| `e2e.yml` | 116 | 1 | ~10 min | PR + Push |
| **Total** | **1102** | **15** | **~75 min** | - |

**Issues:**
- ❌ Sequential execution (jobs wait for each other)
- ❌ Redundant FAISS jobs
- ❌ Runs full suite on every PR (slow feedback)

---

### After (2 workflows, 294 total lines)
| File | Lines | Jobs | Runtime | Runs On |
|------|-------|------|---------|---------|
| `pr-checks.yml` | 124 | 3 parallel | ~3-5 min | Every PR |
| `main-validation.yml` | 170 | 6 parallel | ~15-30 min | Push to main |
| **Total** | **294** | **9** | **~5 min (PR)** | - |

**Improvements:**
- ✅ **73% fewer lines** (1102 → 294)
- ✅ **Parallel execution** (jobs run simultaneously)
- ✅ **Fast PR feedback** (~5 min vs ~75 min)
- ✅ **No redundancy** (consolidated FAISS/dedup/e2e)

---

## 🚀 Benefits

### 1. **Faster PR Feedback** (85% faster)
- Before: Developers wait ~75 min for PR checks
- After: Developers wait ~5 min for PR checks
- **Impact**: 15x faster iteration cycle!

### 2. **Cost Savings (GitHub Actions Minutes)**
- Before: ~75 min per PR + ~75 min per merge = 150 min
- After: ~5 min per PR + ~30 min per merge = 35 min
- **Savings**: 77% fewer CI minutes used

### 3. **Easier Maintenance**
- Before: 1102 lines of YAML with massive inline Python
- After: 294 lines of clean, organized YAML
- **Impact**: 73% less code to maintain

### 4. **Better Organization**
- Clear separation: PR checks vs Main validation
- No confusion about which tests run when
- Easy to disable/enable specific job groups

---

## 🔧 Migration Checklist

- [x] Create `pr-checks.yml`
- [x] Create `main-validation.yml`
- [ ] **Delete old workflows** (ci-cd.yml, dedup_tests.yml, e2e.yml)
- [ ] **Test new workflows**:
  - [ ] Create a test PR → verify pr-checks runs
  - [ ] Merge to main → verify main-validation runs
- [ ] **Monitor first few runs** for any issues
- [ ] Update team documentation (if applicable)

---

## ⚠️ Important Notes

### Test Markers
Make sure your pytest tests use markers for filtering:
```python
@pytest.mark.slow  # Skipped in pr-checks
@pytest.mark.integration  # Runs in main-validation
@pytest.mark.requires_faiss  # Skipped on Windows
```

If tests don't have markers, all tests run (slower but safer).

### FAISS on Windows
Windows jobs skip FAISS tests (`-m "not requires_faiss"`) because FAISS installation on Windows in CI is problematic. Ubuntu jobs run full FAISS tests.

### Coverage Reports
Coverage is uploaded to Codecov only from `main-validation.yml` (Ubuntu job). No coverage on PRs to keep them fast.

---

## 🐛 Troubleshooting

### If PR checks fail unexpectedly:
1. Check test markers are correct
2. Verify imports work (import-check job)
3. Run locally: `pytest tests/ -m "not slow and not integration"`

### If main-validation fails:
1. Check which job failed (check GitHub Actions logs)
2. Run that specific test suite locally
3. Common issues:
   - FAISS not installed (Ubuntu job)
   - Missing test data (integration job)
   - Performance regression (benchmarks)

---

## 📝 Next Steps

1. **Delete old workflows** (backup first if nervous)
2. **Create test PR** to verify pr-checks works
3. **Monitor CI runs** for a few days
4. **Celebrate** 🎉 - You now have enterprise-grade CI/CD!

---

## 🔄 Rollback Plan (If Needed)

If new workflows have issues:
```bash
# Restore old workflows from archive
mv .github/workflows/archive/* .github/workflows/

# Or just disable new ones
mv .github/workflows/pr-checks.yml .github/workflows/pr-checks.yml.disabled
mv .github/workflows/main-validation.yml .github/workflows/main-validation.yml.disabled
```

GitHub will automatically use the old workflows again.
