import json
import os
import time

from .pages.upload_page import UploadPage


def test_cannot_open_empty_collection(page, base_url):
    """Attempting to open an empty collection should redirect to /chat and show an error toast."""
    name = f"empty-coll-{int(time.time())}"

    # Create empty collection via API
    resp = page.request.post(f"{base_url}/api/collections", data={"name": name})
    assert resp.ok
    data = resp.json()
    coll_id = data.get("collection_id") or data.get("id")
    assert coll_id is not None

    # Navigate directly to chat filtered by that empty collection
    page.goto(f"{base_url}/chat?collection={coll_id}")

    # Wait for the redirect back to /chat (collection filter should be cleared)
    page.wait_for_url(f"{base_url}/chat", timeout=5000)

    # URL should not keep the collection query param (should have been replaced)
    assert "collection=" not in page.url

    # Input field should be enabled (not blocked) and placeholder should be default
    input_el = page.locator("[aria-label='Ask a question about your documents']")
    assert input_el.is_enabled()
    placeholder = input_el.get_attribute("placeholder")
    assert "Ask a question" in (placeholder or "")


def test_delete_collection_updates_sidebar(page, base_url):
    """Deleting a collection should update the left sidebar immediately."""
    name = f"temp-coll-{int(time.time())}"

    # Create collection via API (POST JSON body)
    resp = page.request.post(
        f"{base_url}/api/collections",
        headers={"Content-Type": "application/json"},
        data=json.dumps({"name": name}),
    )
    assert resp.ok
    data = resp.json()
    coll_id = data.get("collection_id") or data.get("id")
    assert coll_id is not None

    # Poll the API to ensure the collection is visible server-side (helps diagnose server vs client issues)
    found = False
    for _ in range(20):
        collections_resp = page.request.get(f"{base_url}/api/collections")
        assert collections_resp.ok
        collections = collections_resp.json()
        if any(c.get("name") == name for c in collections):
            found = True
            break
        time.sleep(0.25)
    assert found, "Created collection did not appear in /api/collections"

    # Go to home so sidebar is visible
    page.goto(f"{base_url}/")

    # Trigger a focus event (SWR revalidates on focus) to encourage immediate refresh
    page.evaluate("() => window.dispatchEvent(new Event('focus'))")

    # Wait for the collection to appear in sidebar (poll to be resilient)
    locator = page.locator(f"text={name}")
    found_in_ui = False
    for _ in range(30):
        try:
            if locator.count() > 0:
                found_in_ui = True
                break
        except Exception:
            pass
        time.sleep(0.25)
    assert found_in_ui, f"Collection '{name}' did not appear in sidebar within timeout"
    # Delete the collection via API
    del_resp = page.request.delete(f"{base_url}/api/collections/{coll_id}")
    assert del_resp.ok

    # Poll the API to ensure the collection is removed server-side
    removed = False
    for _ in range(20):
        collections_resp = page.request.get(f"{base_url}/api/collections")
        assert collections_resp.ok
        collections = collections_resp.json()
        if not any(c.get("name") == name for c in collections):
            removed = True
            break
        time.sleep(0.25)
    assert removed, "Deleted collection still present in /api/collections"

    # Sidebar should remove the collection (wait for it to be hidden)
    locator.wait_for(state="hidden", timeout=10000)
    assert locator.count() == 0


def test_add_existing_document_to_collection_via_plus(page, base_url, tmp_path):
    """Click the collection + button and add an existing document from All Files."""
    upload_page = UploadPage(page, base_url)  # use page object from e2e pages

    # Create a small test file and upload via the UI
    f = tmp_path / "e2e_added_doc.txt"
    f.write_text("This document will be used to test adding to a collection via the + modal.")
    test_file = str(f)

    upload_page.goto()
    upload_page.upload_file(test_file)
    assert upload_page.is_file_uploaded_success(timeout=120000), "File upload failed or timed out"

    # Create a collection via UI
    name = f"e2e-coll-{int(time.time())}"
    upload_page.create_collection(name)

    # Wait for collection card to appear
    coll_locator = page.locator(f"text={name}")
    coll_locator.wait_for(timeout=15000)

    # Click the Add (plus) button for this collection
    add_btn = page.locator(f"button[aria-label='Add existing documents to {name}']")
    add_btn.click()

    # Modal should open; select the uploaded file
    # Wait for modal content, then check the document name checkbox and click Add
    doc_checkbox = (
        page.locator(f"text={os.path.basename(test_file)}")
        .locator("..")
        .locator("input[type='checkbox']")
    )
    # Fallback: select by label if structure differs
    if doc_checkbox.count() == 0:
        doc_checkbox = page.locator(f"text={os.path.basename(test_file)}")
    doc_checkbox.click()

    # Confirm add
    page.locator("button:has-text('Add to Collection')").click()

    # Wait for success toast
    page.wait_for_selector("text=Added", timeout=10000)

    # Verify collection count updated (1 document)
    card_count = page.locator(f"text={name}").locator("..").locator("text=1 document")
    assert card_count.count() > 0

    # Cleanup: delete collection via API
    # Retrieve collection id
    resp = page.request.get(f"{base_url}/api/collections")
    assert resp.ok
    cols = resp.json()
    coll = next((c for c in cols if c.get("name") == name), None)
    assert coll is not None
    del_resp = page.request.delete(f"{base_url}/api/collections/{coll.get('id')}")
    assert del_resp.ok
