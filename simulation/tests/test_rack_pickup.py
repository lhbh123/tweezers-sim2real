"""Offline regressions for pickup termination and quaternion conventions."""
import sys
from pathlib import Path
import unittest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from rack_pickup_task import RackPickupConfig, dropped_tool, quat_error, pregrasp_reward_terms

class PickupChecks(unittest.TestCase):
    def test_home_lower_than_rack_is_not_drop(self):
        phase=torch.tensor([0,2,3,4,5,5])
        height=torch.tensor([.9,.9,.9,.9,.9,.73])
        flags=dropped_tool(phase,height-1.,height,.72)
        self.assertEqual(flags.tolist(),[True,True,True,False,False,True])

    def test_quaternion_double_cover_has_same_error(self):
        a=torch.tensor([[.9238795,0.,0.,.3826834]])
        b=torch.tensor([[1.,0.,0.,0.]])
        self.assertTrue(torch.allclose(quat_error(a,b),quat_error(-a,b)))
        self.assertTrue(torch.allclose(quat_error(a,b),quat_error(a,-b)))
        self.assertTrue(torch.allclose(quat_error(a,a),torch.zeros(1,3),atol=1e-6))

    def test_invalid_tolerances_rejected(self):
        for args in [dict(episode_seconds=0),dict(handoff_home_m=float('nan')),
                     dict(decimation=1.5),dict(aperture_response_m=0),dict(face_force_min_N=-1),
                     dict(initial_arm_q_rad=[0]*5),dict(initial_arm_q_rad=[0]*5+[float('nan')]),
                     dict(pregrasp_timeout_s=0),dict(pregrasp_distance_scale_m=0),
                     dict(pregrasp_angle_progress_weight=-1)]:
            with self.subTest(args=args),self.assertRaises(ValueError):RackPickupConfig(**args)

    def test_pregrasp_reward_stationary_cannot_farm_progress(self):
        d=torch.tensor([.3,.1,.0]);a=torch.tensor([.5,.1,.0])
        p,r,cost=pregrasp_reward_terms(d,d,a,a,RackPickupConfig())
        self.assertTrue(torch.equal(p,torch.zeros_like(d)))
        self.assertTrue(torch.equal(r,torch.zeros_like(d)))
        self.assertTrue(bool((cost<=0).all()))
        self.assertLess(float(cost[0]),float(cost[1]))

    def test_pregrasp_progress_rewards_approach_and_cancels_backtracking(self):
        cfg=RackPickupConfig();far=torch.tensor([.5]);near=torch.tensor([.4])
        a=torch.tensor([.3]);b=torch.tensor([.2])
        forward=pregrasp_reward_terms(far,near,a,b,cfg)
        reverse=pregrasp_reward_terms(near,far,b,a,cfg)
        for i in [0,1]:
            self.assertGreater(float(forward[i]),0)
            self.assertLess(float(reverse[i]),0)
            self.assertTrue(torch.allclose(forward[i]+reverse[i],torch.zeros(1)))

    def test_pregrasp_progress_is_independent_of_step_subdivision(self):
        cfg=RackPickupConfig();d=torch.tensor([.4,.3,.2]);a=torch.tensor([.3,.2,.1])
        split=pregrasp_reward_terms(d[:-1],d[1:],a[:-1],a[1:],cfg)
        whole=pregrasp_reward_terms(d[:1],d[-1:],a[:1],a[-1:],cfg)
        for i in [0,1]:self.assertTrue(torch.allclose(split[i].sum(),whole[i][0]))

if __name__=='__main__':unittest.main()
