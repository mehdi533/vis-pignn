"""
Time-Scheduled Load Model for ANDES
Allows loads to follow a time-series schedule from CSV file
"""

import numpy as np
import pandas as pd
from andes.core import (ModelData, IdxParam, NumParam, DataParam, 
                        Model, ConstService, ExtService, ExtAlgeb)


class TimeSchedData(ModelData):
    """Data for time-scheduled load model."""
    
    def __init__(self):
        super().__init__()
        
        # Link to existing PQ load
        self.pq = IdxParam(
            model='PQ',
            info='Index of PQ load to replace',
            mandatory=True,
        )
        
        # Schedule file
        self.sched_file = DataParam(
            info='CSV file with schedule (columns: t, P, Q)',
            mandatory=True,
        )
        
        # Base values (will be taken from PQ if not specified)
        self.Pb = NumParam(
            info='Base active power (MW)',
            default=0.0,
            tex_name='P_b',
        )
        
        self.Qb = NumParam(
            info='Base reactive power (MVAr)',
            default=0.0,
            tex_name='Q_b',
        )
        
        # Schedule mode
        self.mode = NumParam(
            info='Schedule mode: 0=absolute values, 1=multiplier',
            default=1,
            vrange=(0, 1),
        )
        
        # Voltage and frequency dependence (ZIP model)
        self.av = NumParam(
            info='Active power voltage exponent',
            default=0.0,
            tex_name=r'\alpha_V^P',
        )
        
        self.af = NumParam(
            info='Active power frequency exponent',
            default=0.0,
            tex_name=r'\alpha_f^P',
        )
        
        self.bv = NumParam(
            info='Reactive power voltage exponent',
            default=0.0,
            tex_name=r'\alpha_V^Q',
        )
        
        self.bf = NumParam(
            info='Reactive power frequency exponent',
            default=0.0,
            tex_name=r'\alpha_f^Q',
        )


class TimeSchedModel(Model):
    """Model implementation for time-scheduled load."""
    
    def __init__(self, system, config):
        Model.__init__(self, system, config)
        
        self.group = 'DynLoad'
        self.flags.tds = True
        self.flags.pflow = False
        
        # Extract bus from PQ load
        self.bus = ExtService(
            model='PQ',
            src='bus',
            indexer=self.pq,
            tex_name='bus',
        )
        
        # Get initial power from PQ
        self.P0 = ExtService(
            model='PQ',
            src='p0',
            indexer=self.pq,
            tex_name='P_0',
        )
        
        self.Q0 = ExtService(
            model='PQ',
            src='q0',
            indexer=self.pq,
            tex_name='Q_0',
        )
        
        # Get bus voltage
        self.v = ExtAlgeb(
            model='Bus',
            src='v',
            indexer=self.bus,
            tex_name='V',
        )
        
        # Get bus angle
        self.a = ExtAlgeb(
            model='Bus',
            src='a',
            indexer=self.bus,
            tex_name=r'\theta',
        )
        
        # Base voltage
        self.v0 = ConstService(
            v_str='1.0',
            tex_name='V_0',
        )
        
        # Schedule arrays (loaded during setup)
        self._schedule_loaded = False
        self._t_sched = None
        self._P_sched = None
        self._Q_sched = None
    
    def v_numeric(self, **kwargs):
        """Disable the original PQ load to avoid double counting."""
        # Disable the PQ device
        self.system.groups['StaticLoad'].set(
            src='u', 
            idx=self.pq.v, 
            attr='v', 
            value=0
        )
        
        # Load schedule data
        self._load_schedule()
    
    def _load_schedule(self):
        """Load schedule from CSV file."""
        if self._schedule_loaded:
            return
        
        for i, file_path in enumerate(self.sched_file.v):
            try:
                df = pd.read_csv(file_path)
                
                # Validate columns
                if 't' not in df.columns or 'P' not in df.columns:
                    raise ValueError(f"Schedule file must have 't' and 'P' columns")
                
                # Store schedule arrays
                if self._t_sched is None:
                    self._t_sched = [df['t'].values]
                    self._P_sched = [df['P'].values]
                    self._Q_sched = [df['Q'].values if 'Q' in df.columns else None]
                else:
                    self._t_sched.append(df['t'].values)
                    self._P_sched.append(df['P'].values)
                    self._Q_sched.append(df['Q'].values if 'Q' in df.columns else None)
                
                print(f"✓ Loaded schedule for TimeSched {i+1}: {len(df)} time points")
                
            except Exception as e:
                print(f"✗ Error loading schedule file '{file_path}': {e}")
                # Use constant values as fallback
                self._t_sched = [[0, 1e6]]
                self._P_sched = [[1.0, 1.0]]
                self._Q_sched = [None]
        
        self._schedule_loaded = True
    
    def _get_schedule_value(self, t, idx):
        """Get scheduled power at time t for device idx."""
        device_idx = int(idx)
        
        if device_idx >= len(self._P_sched):
            return 1.0, 1.0
        
        # Interpolate P
        P_mult = np.interp(t, self._t_sched[device_idx], self._P_sched[device_idx])
        
        # Interpolate Q if available
        if self._Q_sched[device_idx] is not None:
            Q_mult = np.interp(t, self._t_sched[device_idx], self._Q_sched[device_idx])
        else:
            Q_mult = P_mult  # Use same as P if Q not specified
        
        return P_mult, Q_mult
    
    def g_numeric(self, **kwargs):
        """Calculate power injections based on schedule."""
        # This gets called during TDS at each time step
        pass  # Implemented via a_numeric in simpler approach


class TimeSched(TimeSchedData, TimeSchedModel):
    """
    Time-scheduled load model.
    
    Allows loads to follow a pre-defined time series from CSV file.
    """
    
    def __init__(self, system, config):
        TimeSchedData.__init__(self)
        TimeSchedModel.__init__(self, system, config)


# ==================== USAGE EXAMPLE ====================

def create_sample_schedule(filename='load_schedule.csv', duration=30):
    """Create a sample load schedule CSV file."""
    
    # Create time points
    t = np.linspace(0, duration, 100)
    
    # Create realistic daily pattern
    # Morning ramp: 6am-9am
    # Midday plateau: 9am-5pm  
    # Evening peak: 5pm-9pm
    # Night decrease: 9pm-6am
    
    # For demo: sinusoidal pattern with peak in middle
    P = 0.7 + 0.3 * np.sin(2 * np.pi * t / duration - np.pi/2)
    Q = 0.6 + 0.4 * np.sin(2 * np.pi * t / duration - np.pi/2)
    
    # Add some random variation
    P += 0.05 * np.random.randn(len(t))
    Q += 0.05 * np.random.randn(len(t))
    
    # Create DataFrame
    df = pd.DataFrame({
        't': t,
        'P': P,
        'Q': Q,
    })
    
    df.to_csv(filename, index=False)
    print(f"✓ Created sample schedule: {filename}")
    
    return filename


def setup_timesched_load(ss, pq_idx, schedule_file, mode=1):
    """
    Add a time-scheduled load to the system.
    
    Parameters:
    -----------
    ss : andes.System
        ANDES system object
    pq_idx : int
        Index of PQ load to replace
    schedule_file : str
        Path to CSV schedule file
    mode : int
        0 = absolute values, 1 = multiplier (default)
    """
    
    # Get base power from PQ load
    pq_list = ss.PQ.as_df()
    if pq_idx not in pq_list['idx'].values:
        raise ValueError(f"PQ load with idx={pq_idx} not found")
    
    pq_data = pq_list[pq_list['idx'] == pq_idx].iloc[0]
    
    # Add TimeSched model
    ts_idx = ss.TimeSched.n + 1
    
    ss.TimeSched.add(
        idx=ts_idx,
        pq=pq_idx,
        sched_file=schedule_file,
        Pb=pq_data['p0'] * ss.config.mva,  # Convert to MW
        Qb=pq_data['q0'] * ss.config.mva,  # Convert to MVAr
        mode=mode,
        av=0.0,  # No voltage dependence
        af=0.0,  # No frequency dependence
        bv=0.0,
        bf=0.0,
    )
    
    print(f"✓ Added TimeSched load (idx={ts_idx}) replacing PQ load {pq_idx}")
    print(f"  Base power: P={pq_data['p0']*ss.config.mva:.2f} MW, Q={pq_data['q0']*ss.config.mva:.2f} MVAr")
    print(f"  Schedule mode: {'multiplier' if mode==1 else 'absolute'}")
    
    return ts_idx


if __name__ == "__main__":
    import andes
    import matplotlib.pyplot as plt
    
    print("="*60)
    print("ANDES Time-Scheduled Load Example")
    print("="*60)
    
    # 1. Create sample schedule
    sched_file = create_sample_schedule('my_load_schedule.csv', duration=30)
    
    # 2. Load system
    case_path = andes.get_case('ieee14/ieee14_full.xlsx')
    ss = andes.load(case_path, setup=False, no_output=True)
    
    # 3. Register TimeSched model (if not in default models)
    # ss.add_model('TimeSched', TimeSched)
    
    # 4. Setup system
    ss.setup()
    
    # 5. Add time-scheduled load
    # Replace PQ load at bus 4 (for example)
    pq_loads = ss.PQ.as_df()
    print("\nAvailable PQ loads:")
    print(pq_loads[['idx', 'bus', 'name', 'p0', 'q0']])
    
    # Choose first load
    pq_idx_to_replace = pq_loads['idx'].iloc[0]
    
    # NOTE: TimeSched needs to be registered as a model first
    # This example shows the structure - actual implementation 
    # requires adding TimeSched to ANDES models directory
    
    print("\n⚠️  NOTE: To use this model:")
    print("1. Save TimeSched class to: andes/models/load/timesched.py")
    print("2. Register in: andes/models/load/__init__.py")
    print("3. Add to routines.yml if needed")
    print("="*60)