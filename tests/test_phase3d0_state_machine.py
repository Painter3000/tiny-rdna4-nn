import subprocess,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
class StateMachine(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.bin=ROOT/"build_phase3d0/state_machine_probe"
  subprocess.run(["g++","-std=c++17",str(ROOT/"src/impl/phase3d_inprocess_loop.cpp"),str(ROOT/"tests/state_machine_probe.cpp"),"-o",str(cls.bin)],check=True)
 def test_valid_four_steps(self):
  p=subprocess.run([str(self.bin)],capture_output=True,text=True);self.assertEqual(0,p.returncode);self.assertIn("PASS",p.stdout)
 def test_illegal_transition_fails_closed(self):
  source='''#include "src/impl/phase3d_inprocess_loop.hpp"\nint main(){p3d0::StateMachine s;try{s.transition(p3d0::LoopState::ADAM_COMPLETE,1);}catch(...){return s.state()==p3d0::LoopState::FAILED?0:2;}return 3;}'''
  p=ROOT/"build_phase3d0/illegal.cpp";p.write_text(source)
  b=ROOT/"build_phase3d0/illegal_probe";subprocess.run(["g++","-std=c++17","-I",str(ROOT),str(ROOT/"src/impl/phase3d_inprocess_loop.cpp"),str(p),"-o",str(b)],check=True)
  self.assertEqual(0,subprocess.run([str(b)]).returncode)
