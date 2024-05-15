import requests
import hashlib
import os
from fastapi import HTTPException

CVSS_API_ENDPOINT = os.getenv('CVSS_API_ENDPOINT')
CVSS_API_KEY = os.getenv('CVSS_API_KEY')
CVSS_API_SECRET = os.getenv('CVSS_API_SECRET')

def get_video_token(course_id: int, chapter_id: int):
    payload = {
        "courseId": course_id,
        "chapterId": chapter_id,
    }

    digest = create_digest(course_id, chapter_id)

    response = requests.post(
        f"{CVSS_API_ENDPOINT}/api/v1/auth/token",
        json=payload,
        headers={
            'resourceOwnerId': CVSS_API_KEY,
            'X-Skilllane-Biscuit': digest,
        },
    )
    
    if response.status_code != 200:
        raise HTTPException(status_code=500, detail="Cannot get video token")
    
    response_json = response.json()

    return response_json["token"]

def get_video_url(course_id: int, chapter_id: int, video_token: str, video_version: str):
    videoEndPoint = CVSS_API_ENDPOINT
    queryParams = f"v={video_version}&token={video_token}" if video_version != 0 else f"token={video_token}"
    return f"{videoEndPoint}/api/v1/playlist/{course_id}/chapters/{chapter_id}.m3u8?{queryParams}"

def create_digest(course_id: int, chapter_id: int):
    presigned = f"{chapter_id}{course_id}{CVSS_API_SECRET}"
    digestHash = hashlib.sha256(presigned.encode()).hexdigest()
    return digestHash