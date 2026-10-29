"""Offline checks for bounded sampling and physically consistent camera chains."""
import copy
import json
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'simulation/src'))
from episode_randomization import local_delta,perturb_camera_rig,sample_episode


class RandomizationTest(unittest.TestCase):
    def setUp(self):
        self.profile=json.loads((ROOT/'simulation/configs/randomized_collection_v1.json').read_text())
        self.layout=json.loads((ROOT/'simulation/configs/calibrated_demo_y_table.json').read_text())
        self.hand=list(json.loads((ROOT/'simulation/configs/xhand_local.json').read_text())['q_reference'])
        self.rig=json.loads((ROOT/'calibration/results/camera_rig_isaac.json').read_text())

    def test_seed_reproducible_and_changes_all_sample_categories(self):
        a=sample_episode(self.profile,self.layout,self.hand,1)
        self.assertEqual(a,sample_episode(self.profile,self.layout,self.hand,1))
        b=sample_episode(self.profile,self.layout,self.hand,2)
        for key in ['start_xy_m','goal_xy_m','home_offset_xy_m','home_height_offset_m',
                    'tool_yaw_offset_deg','hand_target_offsets_rad','cameras']:
            self.assertNotEqual(a[key],b[key],key)

    def test_sampling_bounds_and_separation(self):
        for seed in range(100):
            c=sample_episode(self.profile,self.layout,self.hand,seed)
            for key,lo,hi in [('start_xy_m','start_xy_lower_m','start_xy_upper_m'),
                              ('goal_xy_m','goal_xy_lower_m','goal_xy_upper_m')]:
                self.assertTrue(np.all(np.asarray(c[key])>=self.profile[lo]))
                self.assertTrue(np.all(np.asarray(c[key])<=self.profile[hi]))
            self.assertGreaterEqual(np.linalg.norm(np.asarray(c['start_xy_m'])-c['goal_xy_m']),.25)
            self.assertLessEqual(max(abs(v) for v in c['hand_target_offsets_rad'].values()),.001)

    def test_camera_perturbation_preserves_link_tcp_and_original(self):
        original=copy.deepcopy(self.rig)
        # Include a non-identity link/TCP relationship to catch a composition error.
        fixed=local_delta([.01,-.02,.03],[3,2,1])
        self.rig['cameras']['wrist']['T_link_Cw']=(fixed@np.asarray(self.rig['cameras']['wrist']['T_TCP_Cw'])).tolist()
        before=copy.deepcopy(self.rig)
        settings=sample_episode(self.profile,self.layout,self.hand,1)['cameras']
        changed=perturb_camera_rig(self.rig,settings)
        wrist=changed['cameras']['wrist']
        self.assertTrue(np.allclose(np.asarray(wrist['T_link_Cw'])@np.linalg.inv(wrist['T_TCP_Cw']),fixed))
        self.assertEqual(self.rig,before)
        for name,key in [('wrist','T_TCP_Cw'),('global','T_B_Cg')]:
            T=np.asarray(changed['cameras'][name][key]);R=T[:3,:3]
            self.assertTrue(np.allclose(R.T@R,np.eye(3)))
            self.assertAlmostEqual(np.linalg.det(R),1)
            self.assertFalse(np.allclose(T,original['cameras'][name][key]))
            self.assertEqual(changed['cameras'][name]['rectified'],original['cameras'][name]['rectified'])

    def test_reject_excessive_hand_offsets(self):
        self.profile['hand_target_radius_rad']=.1
        with self.assertRaisesRegex(ValueError,'hand offsets'):
            sample_episode(self.profile,self.layout,self.hand,1)

    def test_reject_unreachable_sampling_distance(self):
        self.profile['minimum_transfer_m']=10
        with self.assertRaisesRegex(ValueError,'transfer distance'):
            sample_episode(self.profile,self.layout,self.hand,1)

    def test_reject_samples_outside_table(self):
        self.profile['start_xy_lower_m']=[1,1]
        self.profile['start_xy_upper_m']=[2,2]
        with self.assertRaisesRegex(ValueError,'tabletop'):
            sample_episode(self.profile,self.layout,self.hand,1)


if __name__=='__main__':unittest.main()
