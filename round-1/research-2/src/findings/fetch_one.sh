#!/bin/bash
idx="$1"; url="$2"
SKILL_DIR=/home/adrian/projects/ai-inventor-wt-integ/.claude/skills/aii-web-tools
PY="$SKILL_DIR/../.ability_client_venv/bin/python"
out=/home/adrian/projects/ai-inventor-wt-integ/aii_data/users/admin/runs/run_Ymcqd66mFtJZ/3_invention_loop/iter_1/gen_art/gen_art_research_2/findings/fetched/$idx.txt
timeout 300 $PY "$SKILL_DIR/scripts/aii_fast_web_fetch.py" fetch --url "$url" --max-chars 2000000 > "$out" 2>&1
echo "$idx $? $(wc -c < "$out")" >> /home/adrian/projects/ai-inventor-wt-integ/aii_data/users/admin/runs/run_Ymcqd66mFtJZ/3_invention_loop/iter_1/gen_art/gen_art_research_2/findings/fetched/_status.log
