import os

def check_frontend_clean():
    errors = []
    
    # 1. Check if NotionPromptForm is gone
    if os.path.exists("frontend/components/notion-prompt-form.tsx"):
        errors.append("frontend/components/notion-prompt-form.tsx still exists")
        
    # 2. Check chat page
    with open("frontend/app/chat/page.tsx", "r") as f:
        content = f.read()
        if "NotionPromptForm" in content:
            errors.append("NotionPromptForm still referenced in chat/page.tsx")
            
    # 3. Check upload page
    with open("frontend/app/upload/page.tsx", "r") as f:
        content = f.read()
        if "initialCategories" in content:
            errors.append("initialCategories still in upload/page.tsx")
        if "SAMPLE_DATA" in content:
            errors.append("SAMPLE_DATA still in upload/page.tsx")
            
    if errors:
        print("❌ Frontend cleanup failed:")
        for error in errors:
            print(f"  - {error}")
        exit(1)
    else:
        print("✅ Frontend cleanup verified!")
        exit(0)

if __name__ == "__main__":
    check_frontend_clean()
