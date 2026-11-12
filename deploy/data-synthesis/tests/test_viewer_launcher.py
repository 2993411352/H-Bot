import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch, MagicMock


@unittest.skipUnless(sys.platform.startswith('linux'), 'Host launcher targets native Linux')
class ViewerLauncherTests(unittest.TestCase):
    def test_viewer_needs_no_gpu_or_asset_access(self):
        path = Path(__file__).resolve().parents[1]/'manage.py'
        spec = importlib.util.spec_from_file_location('synthesis_manage',path)
        mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); work = root/'work'; work.mkdir()
            config = root/'local.json'
            config.write_text(json.dumps(dict(work=str(work),assets=str(root/'missing-assets'),
                reference=str(root/'missing-reference'),image_prefix='hbot/test',gpu=0,accept_eula=False)))
            with patch.object(sys,'argv',['manage.py','--config',str(config),'viewer','--workers','101','102']), \
                 patch.object(mod,'read_gpu',side_effect=AssertionError('Must not use GPU')), \
                 patch.object(mod.subprocess,'run',return_value=MagicMock(returncode=0)) as run:
                mod.main()
            command = run.call_args_list[-1].args[0]
            self.assertNotIn('--gpus',command)
            self.assertIn(f'type=bind,src={work},dst=/source,readonly',command)
            self.assertIn('NVIDIA_VISIBLE_DEVICES=void',command)
            self.assertEqual(command[-2:],['--port','18765'])
            self.assertFalse((root/'missing-assets').exists())


if __name__ == '__main__': unittest.main()
