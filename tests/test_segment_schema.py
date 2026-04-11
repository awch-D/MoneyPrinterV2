import json
import os
import sys
import unittest

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from novel.segment_schema import parse_chapter_plan_json


class SegmentSchemaTests(unittest.TestCase):
    def test_parse_plain_json(self) -> None:
        raw = json.dumps(
            {
                "style_bible": "Film noir, high contrast, 35mm.",
                "characters": [{"id": "c1", "name": "Ann", "look": "Short black hair, red coat."}],
                "segments": [
                    {
                        "narration": "She opened the door.",
                        "scene_summary": "Door",
                        "image_prompt": "Medium shot, door opening, dramatic light.",
                        "visible_character_ids": ["c1"],
                    },
                    {
                        "narration": "Rain hit the pavement.",
                        "scene_summary": "Street",
                        "image_prompt": "Wide shot, wet street reflections.",
                        "visible_character_ids": [],
                    },
                ],
            }
        )
        plan = parse_chapter_plan_json(raw)
        self.assertEqual(len(plan.segments), 2)
        self.assertEqual(plan.segments[0].narration, "She opened the door.")
        self.assertEqual(plan.characters[0].id, "c1")

    def test_brace_extractor_handles_brace_inside_string(self) -> None:
        """Regression: naive rfind('}') breaks when narration contains a brace."""
        inner = {
            "style_bible": "Noir.",
            "characters": [],
            "segments": [
                {
                    "narration": 'He whispered "stop}" and froze.',
                    "scene_summary": "x",
                    "image_prompt": "Close-up, tense face.",
                    "visible_character_ids": [],
                },
                {
                    "narration": "End.",
                    "scene_summary": "y",
                    "image_prompt": "Wide exterior night.",
                    "visible_character_ids": [],
                },
            ],
        }
        raw = "Prefix noise " + json.dumps(inner) + " trailing"
        plan = parse_chapter_plan_json(raw)
        self.assertIn("stop}", plan.segments[0].narration)

    def test_parse_fenced_json(self) -> None:
        inner = {
            "style_bible": "Anime pastel colors.",
            "characters": [],
            "segments": [
                {
                    "narration": "Hello.",
                    "scene_summary": "x",
                    "image_prompt": "A character waves.",
                    "visible_character_ids": [],
                },
                {
                    "narration": "Goodbye.",
                    "scene_summary": "y",
                    "image_prompt": "Sunset horizon.",
                    "visible_character_ids": [],
                },
                {
                    "narration": "Third.",
                    "scene_summary": "z",
                    "image_prompt": "Night city.",
                    "visible_character_ids": [],
                },
            ],
        }
        raw = "Here is JSON:\n```json\n" + json.dumps(inner) + "\n```"
        plan = parse_chapter_plan_json(raw)
        self.assertEqual(len(plan.segments), 3)


if __name__ == "__main__":
    unittest.main()
