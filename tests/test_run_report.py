import sys
from pathlib import Path
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'analysis'))
from build_run_report import polish_text


class ReportTests(unittest.TestCase):
    def test_prose_spacing_preserves_data_and_code(self):
        source='<style>color:#a5bcde</style><p>All16 joints at5N with8Nm</p><img src="data:image/webp;base64,All16ab5N"><code>a5b8Nm</code><script>const a5=16;</script>'
        result=polish_text(source)
        self.assertIn('All 16 joints at 5 N with 8 Nm',result)
        self.assertIn('base64,All16ab5N',result)
        self.assertIn('<code>a5b8Nm</code>',result)
        self.assertIn('const a5=16;',result)
        self.assertIn('color:#a5bcde',result)

    def test_entities_and_document_structure_preserved(self):
        source='<!doctype html><html><body><p>A &amp; B &#8805; 2s</p></body></html>'
        self.assertEqual(polish_text(source),'<!doctype html><html><body><p>A &amp; B &#8805; 2 s</p></body></html>')


if __name__=='__main__':unittest.main()
