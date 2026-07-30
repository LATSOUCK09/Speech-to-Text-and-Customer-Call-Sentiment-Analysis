curl -X POST "http://127.0.0.1:8000/analyze" \
  -F "file=@audio/ElevenLabs_2026-07-28T17_47_35_Mélanie - Storyteller_pvc_sp115_s53_sb60_se4_b_m2.mp3" \
  -F "chunk_duration=20" \
  -F "overlap=2"
