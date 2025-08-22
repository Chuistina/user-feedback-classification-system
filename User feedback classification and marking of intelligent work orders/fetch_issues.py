import requests
import json
import os

def fetch_github_issues(owner, repo, output_dir="data/raw", per_page=100, max_pages=5):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    all_issues = []
    page = 1
    while page <= max_pages:
        url = f"https://api.github.com/repos/{owner}/{repo}/issues"
        params = {
            "state": "all",  # Fetch all issues (open and closed)
            "per_page": per_page,
            "page": page
        }
        headers = {
            "Accept": "application/vnd.github.v3+json",
            "Authorization": "Bearer YOUR_GITHUB_TOKEN_HERE"
        }

        response = requests.get(url, params=params, headers=headers)

        if response.status_code == 200:
            issues = response.json()
            if not issues:
                break  # No more issues to fetch
            all_issues.extend(issues)
            print(f"Fetched {len(issues)} issues from page {page}")
            page += 1
        else:
            print(f"Error fetching issues: {response.status_code} - {response.text}")
            break

    output_file = os.path.join(output_dir, f"{owner}_{repo}_issues.jsonl")
    with open(output_file, "w", encoding="utf-8") as f:
        for issue in all_issues:
            f.write(json.dumps(issue, ensure_ascii=False) + "\n")
    print(f"Saved {len(all_issues)} issues to {output_file}")

if __name__ == "__main__":
    # Example usage: Replace with a real GitHub repository
    # For demonstration, I'll use a well-known open-source project with many issues.
    # It's important to choose a repository that is actively maintained and has a good number of issues.
    # Let's try 'tensorflow/tensorflow' or 'microsoft/vscode'
    # Due to rate limits and large data, I'll start with a smaller, but active repo for testing.
    # For example, 'pallets/flask' or 'requests/requests'
    owner = "requests"
    repo = "requests"
    fetch_github_issues(owner, repo, max_pages=2) # Fetching a few pages for testing


