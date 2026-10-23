"""Read collider identities from native PhysX contact reports, without force writes."""
from pathlib import Path
import json
import numpy as np


class ContactShapeRecorder:
    def __init__(self, scene, output):
        import carb.settings
        from omni.physx import get_physx_simulation_interface
        from pxr import PhysicsSchemaTools
        self.scene = scene
        self.output = Path(output)
        self.stream = self.output.open('w')
        self.path_of = lambda value: str(PhysicsSchemaTools.intToSdfPath(value))
        self.time_s = 0.
        self.pending = []
        self.pair_totals = {}
        self.report_count = 0
        self.native_calls = 0
        self.native_header_count = 0
        self.native_data_count = 0
        self.empty_native_steps = 0
        self.rejected_header_count = 0
        self.header_samples = []
        self._sampled_pairs = set()
        self.step_samples = []
        self.error_count = 0
        self.errors = []
        self.disable_contact_processing = carb.settings.get_settings().get_as_bool('/physics/disableContactProcessing')
        # Tensor-only headless stepping does not necessarily dispatch subscribed
        # application events. Read the completed physics step synchronously.
        self.interface = get_physx_simulation_interface()
        self.attached_stage_id = int(self.interface.get_attached_stage())

    def _error(self, operation, error):
        self.error_count += 1
        if len(self.errors) < 16:
            self.errors.append({'time_s': self.time_s, 'operation': operation,
                                'type': type(error).__name__, 'message': str(error)})

    def _report(self, headers, data):
        for h in headers:
            paths = [self.path_of(h.collider0), self.path_of(h.collider1)]
            actors = [self.path_of(h.actor0), self.path_of(h.actor1)]
            tool = [i for i, p in enumerate(paths) if '/Tweezers/' in p]
            robot = [i for i, p in enumerate(paths) if '/Robot/' in p]
            key = tuple(actors + paths)
            if key not in self._sampled_pairs and len(self.header_samples) < 32:
                self._sampled_pairs.add(key)
                self.header_samples.append({'time_s': self.time_s,
                    'actor_paths': actors, 'collider_paths': paths,
                    'actor_ids': [int(h.actor0), int(h.actor1)],
                    'collider_ids': [int(h.collider0), int(h.collider1)],
                    'stage_id': int(h.stage_id), 'event_type': str(h.type),
                    'contact_data_offset': int(h.contact_data_offset),
                    'num_contact_data': int(h.num_contact_data),
                    'passes_target_path_filter': len(tool) == 1 and len(robot) == 1})
            if len(tool) != 1 or len(robot) != 1:
                self.rejected_header_count += 1
                continue
            rp, tp = paths[robot[0]], paths[tool[0]]
            finger = next((f for f in ('thumb','index','middle','ring','pinky') if 'right_'+f+'_' in rp), 'other')
            kind = 'skin' if 'elastomer' in rp.lower() else 'non_skin'
            points = []
            normal_impulse = 0.
            for j in range(h.contact_data_offset, h.contact_data_offset+h.num_contact_data):
                d = data[j]
                impulse = np.asarray(d.impulse, dtype=float)
                normal = np.asarray(d.normal, dtype=float)
                magnitude = abs(float(impulse)) if impulse.ndim == 0 else abs(float(impulse @ normal))
                normal_impulse += magnitude
                points.append({'position_w_m': list(d.position), 'normal_w': list(d.normal),
                    'impulse_raw': impulse.tolist(), 'separation_m': float(d.separation),
                    'normal_impulse_magnitude_Ns': magnitude})
            row = {'robot_collider': rp, 'tool_collider': tp, 'finger': finger,
                   'surface_class': kind, 'event_type': str(h.type), 'points': points,
                   'normal_impulse_magnitude_Ns': normal_impulse}
            self.pending.append(row)
            key = (rp, tp, finger, kind)
            total = self.pair_totals.setdefault(key, {'reports': 0, 'points': 0, 'normal_impulse_magnitude_Ns': 0.})
            total['reports'] += 1
            total['points'] += len(points)
            total['normal_impulse_magnitude_Ns'] += normal_impulse
            self.report_count += 1

    def record_step(self, dt):
        self.time_s += float(dt)
        self.native_calls += 1
        try:
            headers, data = self.interface.get_contact_report()
            self.native_header_count += len(headers)
            self.native_data_count += len(data)
            self.empty_native_steps += int(len(headers) == 0)
            if self.native_calls <= 16 or self.native_calls % 60 == 0:
                self.step_samples.append({'time_s': self.time_s,
                                          'header_count': len(headers), 'data_count': len(data)})
            self._report(headers, data)
        except Exception as error:
            self._error('get_contact_report_or_decode', error)
        for row in self.pending:
            row['time_s'] = self.time_s
            row['physics_dt_s'] = float(dt)
            self.stream.write(json.dumps(row, ensure_ascii=False)+'\n')
        self.pending.clear()

    def close(self):
        import carb.settings
        self.stream.close()
        summary = {'method': 'Native PhysX get_contact_report after each physics step; collider identities and reported impulses.',
                   'runtime_object_state_writes': False, 'report_count': self.report_count,
                   'duration_s': self.time_s, 'complete': not bool(self.pending),
                   'disable_contact_processing': self.disable_contact_processing,
                   'disable_contact_processing_at_close': carb.settings.get_settings().get_as_bool('/physics/disableContactProcessing'),
                   'attached_stage_id': self.attached_stage_id,
                   'native_calls': self.native_calls,
                   'native_header_count_before_filter': self.native_header_count,
                   'native_data_count_before_filter': self.native_data_count,
                   'empty_native_steps': self.empty_native_steps,
                   'rejected_header_count': self.rejected_header_count,
                   'native_report_available': self.native_header_count > 0,
                   'target_shape_records_available': self.report_count > 0,
                   'error_count': self.error_count, 'errors': self.errors,
                   'unfiltered_header_samples': self.header_samples,
                   'native_step_samples': self.step_samples,
                   'pairs': [dict(robot_collider=k[0], tool_collider=k[1], finger=k[2],
                                  surface_class=k[3], **v) for k,v in self.pair_totals.items()],
                   'scope': 'Robot-to-tweezers contact events only. Compare impulses/dt against independent tensor reader before treating aggregated values as a complete force measurement.'}
        self.output.with_suffix('.summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n')
        return summary
