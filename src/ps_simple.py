import andes
import numpy as np
import matplotlib
import networkx as nx
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import json
import os
import xlsxwriter

if os.environ.get("MPLBACKEND", "").lower() == "":
    try:
        matplotlib.use("TkAgg")      # or "Qt5Agg" if you have PyQt5
    except Exception:
        pass

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle, Polygon

class PowerSystem:
    """
    Comprehensive power system class based on ANDES for GUI integration,
    visualization, and dynamic network modification.
    """
    
    def __init__(self, case_path: Optional[str] = None):
        """
        Initialize power system with optional case file.
        
        Args:
            case_path: Path to ANDES case file (e.g., 'kundur_full.xlsx', 'ieee39.xlsx')
        """
        self.sys = andes.System()
        self.case_path = case_path
        self.graph = None  # NetworkX graph for visualization
        self.positions = {}  # Node positions for plotting
        
        if case_path:
            self.load_case(case_path)

        print("Power system initiated!")
    
    # ========================== LOADING & SAVING ==========================
    
    def load_case(self, case_path: str, addfile: Optional[str] = None, setup: bool = False):
        """
        Load a case into self.sys.
        - If addfile is given or case_path endswith .raw, we assume RAW+DYR.
        - Otherwise we assume XLSX.
        By default we load with setup=False so we can still inject events.
        """
        try:
            def _resolve(p):
                # allow either absolute path or ANDES built-in case name
                return p if os.path.exists(p) else andes.get_case(p)

            # RAW + DYR case
            if addfile is not None or case_path.lower().endswith(".raw"):
                raw_path = _resolve(case_path)
                if addfile is None:
                    # guess matching .dyr in same folder
                    dyr_guess = os.path.splitext(raw_path)[0] + ".dyr"
                    addfile_path = _resolve(dyr_guess)
                else:
                    addfile_path = _resolve(addfile)

                self.sys = andes.load(raw_path, addfile=addfile_path, setup=setup)
                self.case_path = f"{raw_path} + {addfile_path}"

            # XLSX case
            else:
                xlsx_path = _resolve(case_path)
                # Note: andes.load(xlsx_path, setup=False) also works for xlsx in recent ANDES.
                self.sys = andes.load(xlsx_path, setup=setup)
                self.case_path = xlsx_path

            # No graph yet if we haven't called setup() (buses/lines exist, that's enough)
            self._build_graph()

            print(f"✓ Case loaded (setup={setup}).")
            return True

        except Exception as e:
            print(f"✗ Error loading case: {e}")
            return False

    def save_case_xlsxwriter(self, output_path: str) -> bool:
        """
        Export current system snapshot to an ANDES-like Excel workbook using xlsxwriter.
        Sheets: Summary, Bus, Line, Slack, PV, PQ.
        Note: This is a readable export for class/labs — not guaranteed round-trip loadable.
        """
        try:
            # --- Build data frames from ANDES system ---
            # Bus
            bus_cols = ["idx","name","Vn","v0","a0","area","zone"]
            df_bus = pd.DataFrame(self._get_bus_data())
            if not df_bus.empty:
                # ensure columns order & fill missing with defaults
                for c in bus_cols:
                    if c not in df_bus.columns:
                        df_bus[c] = {"area":1, "zone":1}.get(c, None)
                df_bus = df_bus[bus_cols]

            # Line
            line_cols = ["idx","name","bus1","bus2","r","x","b","rate_a"]
            df_line = pd.DataFrame(self._get_line_data())
            if not df_line.empty:
                for c in line_cols:
                    if c not in df_line.columns:
                        df_line[c] = 0
                df_line = df_line[line_cols]

            # Slack (if available via PV/Slack models; here we try to extract a Slack bus row)
            slack_cols = ["idx","name","bus","v0","a0","Sn","u"]
            df_slack = pd.DataFrame(columns=slack_cols)
            if hasattr(self.sys, "Slack"):
                sl = self.sys.Slack
                if len(sl.idx.v):
                    df_slack = pd.DataFrame({
                        "idx":  sl.idx.v.copy(),
                        "name": getattr(sl, "name", sl.idx).v if hasattr(sl, "name") else [f"Slack_{i}" for i in sl.idx.v],
                        "bus":  sl.bus.v.copy(),
                        "v0":   sl.v0.v.copy() if hasattr(sl, "v0") else 1.0,
                        "a0":   sl.a0.v.copy() if hasattr(sl, "a0") else 0.0,
                        "Sn":   sl.Sn.v.copy() if hasattr(sl, "Sn") else 100.0,
                        "u":    sl.u.v.copy()  if hasattr(sl, "u")  else 1.0,
                    })

            # PV generators (steady-state)
            pv_cols = ["idx","name","bus","p0","v0","qmax","qmin","Sn","u"]
            df_pv = pd.DataFrame(columns=pv_cols)
            if hasattr(self.sys, "PV"):
                pv = self.sys.PV
                if len(pv.idx.v):
                    df_pv = pd.DataFrame({
                        "idx":  pv.idx.v.copy(),
                        "name": getattr(pv, "name", pv.idx).v if hasattr(pv, "name") else [f"PV_{i}" for i in pv.idx.v],
                        "bus":  pv.bus.v.copy(),
                        "p0":   pv.p0.v.copy() if hasattr(pv, "p0") else 0.0,
                        "v0":   pv.v0.v.copy() if hasattr(pv, "v0") else 1.0,
                        "qmax": pv.qmax.v.copy() if hasattr(pv, "qmax") else 0.3,
                        "qmin": pv.qmin.v.copy() if hasattr(pv, "qmin") else -0.3,
                        "Sn":   pv.Sn.v.copy() if hasattr(pv, "Sn") else 100.0,
                        "u":    pv.u.v.copy()  if hasattr(pv, "u")  else 1.0,
                    })
                    df_pv = df_pv[pv_cols]

            # PQ loads
            pq_cols = ["idx","name","bus","p0","q0","u"]
            df_pq = pd.DataFrame(columns=pq_cols)
            if hasattr(self.sys, "PQ"):
                pq = self.sys.PQ
                if len(pq.idx.v):
                    df_pq = pd.DataFrame({
                        "idx":  pq.idx.v.copy(),
                        "name": getattr(pq, "name", pq.idx).v if hasattr(pq, "name") else [f"Load_{i}" for i in pq.idx.v],
                        "bus":  pq.bus.v.copy(),
                        "p0":   pq.p0.v.copy(),
                        "q0":   pq.q0.v.copy(),
                        "u":    pq.u.v.copy() if hasattr(pq, "u") else 1.0,
                    })
                    df_pq = df_pq[pq_cols]

            # Summary
            summary = pd.DataFrame([{
                "case":        os.path.basename(self.case_path) if self.case_path else "Custom",
                "num_buses":   (len(df_bus)  if not df_bus.empty  else 0),
                "num_lines":   (len(df_line) if not df_line.empty else 0),
                "num_slack":   (len(df_slack) if not df_slack.empty else 0),
                "num_pv":      (len(df_pv)  if not df_pv.empty  else 0),
                "num_pq":      (len(df_pq)  if not df_pq.empty  else 0),
                "base_mva":    getattr(self.sys.config, "mva", 100.0),
                "frequency":   getattr(self.sys.config, "freq", 50.0),
            }])

            # --- Write with xlsxwriter ---
            with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
                # write all sheets (skip empties)
                summary.to_excel(writer, sheet_name="Summary", index=False)
                if not df_bus.empty:  df_bus.to_excel(writer, sheet_name="Bus", index=False)
                if not df_line.empty: df_line.to_excel(writer, sheet_name="Line", index=False)
                if not df_slack.empty: df_slack.to_excel(writer, sheet_name="Slack", index=False)
                if not df_pv.empty:   df_pv.to_excel(writer, sheet_name="PV", index=False)
                if not df_pq.empty:   df_pq.to_excel(writer, sheet_name="PQ", index=False)

                wb = writer.book
                hdr = wb.add_format({"bold": True, "bg_color": "#F2F2F2", "border": 1})
                cel = wb.add_format({"border": 1})
                num = wb.add_format({"border": 1, "num_format": "0.0000"})
                intf = wb.add_format({"border": 1, "num_format": "0"})

                def _format_sheet(name, df):
                    ws = writer.sheets[name]
                    # header and freeze/filter
                    ws.set_row(0, None, hdr)
                    ws.freeze_panes(1, 0)
                    ws.autofilter(0, 0, max(0, len(df)), max(0, len(df.columns)-1))

                    # auto width + numeric formatting heuristics
                    for i, col in enumerate(df.columns):
                        # width by max string length (+2 padding)
                        try:
                            width = max(10, min(40, int(df[col].astype(str).map(len).max()) + 2))
                        except Exception:
                            width = 12
                        # choose format
                        fmt = cel
                        if pd.api.types.is_integer_dtype(df[col]): fmt = intf
                        elif pd.api.types.is_float_dtype(df[col]): fmt = num
                        ws.set_column(i, i, width, fmt)

                _format_sheet("Summary", summary)
                if not df_bus.empty:  _format_sheet("Bus",   df_bus)
                if not df_line.empty: _format_sheet("Line",  df_line)
                if not df_slack.empty:_format_sheet("Slack", df_slack)
                if not df_pv.empty:   _format_sheet("PV",    df_pv)
                if not df_pq.empty:   _format_sheet("PQ",    df_pq)

            print(f"✓ Wrote ANDES-like workbook to: {output_path}")
            return True

        except Exception as e:
            print(f"✗ Error writing xlsx: {e}")
            return False

    # ========================== DATA ACCESS ==========================
    
    def _get_bus_data(self) -> List[Dict]:
        """Get all bus data as list of dictionaries."""
        buses = []
        for i, idx in enumerate(self.sys.Bus.idx.v):
            buses.append({
                'idx': idx,
                'name': self.sys.Bus.name.v[i] if hasattr(self.sys.Bus, 'name') else f"Bus_{idx}",
                'Vn': self.sys.Bus.Vn.v[i],
                'v0': self.sys.Bus.v0.v[i] if hasattr(self.sys.Bus, 'v0') else 1.0,
                'a0': self.sys.Bus.a0.v[i] if hasattr(self.sys.Bus, 'a0') else 0.0,
                'area': self.sys.Bus.area.v[i] if hasattr(self.sys.Bus, 'area') else 1,
                'zone': self.sys.Bus.zone.v[i] if hasattr(self.sys.Bus, 'zone') else 1,
            })
        return buses
    
    def _get_generator_data(self) -> List[Dict]:
        """Get all generator data."""
        gens = []
        # Check for different generator models
        gen_models = ['GENROU', 'REGCA1']
        
        for model_name in gen_models:
            if hasattr(self.sys, model_name):
                model = getattr(self.sys, model_name)
                for i, idx in enumerate(model.idx.v):
                    gens.append({
                        'idx': idx,
                        'type': model_name,
                        'bus': model.bus.v[i],
                        'name': model.name.v[i] if hasattr(model, 'name') else f"{model_name}_{idx}",
                        'Sn': model.Sn.v[i] if hasattr(model, 'Sn') else 100,
                        'p0': model.p0.v if hasattr(model, 'p0') else 0, # TODO: fix because sometimes comes as list, otherwise as float
                        'v0': model.v0.v[i] if hasattr(model, 'v0') else 1.0,
                        # TODO: also add gen for REGCA1 generators for example, maybe use a built in function instead to do that
                    })
        return gens
    
    def _get_line_data(self) -> List[Dict]:
        """Get all transmission line data."""
        lines = []
        for i, idx in enumerate(self.sys.Line.idx.v):
            lines.append({
                'idx': idx,
                'name': self.sys.Line.name.v[i] if hasattr(self.sys.Line, 'name') else f"Line_{idx}",
                'bus1': self.sys.Line.bus1.v[i],
                'bus2': self.sys.Line.bus2.v[i],
                'r': self.sys.Line.r.v[i],
                'x': self.sys.Line.x.v[i],
                'b': self.sys.Line.b.v[i] if hasattr(self.sys.Line, 'b') else 0,
                'rate_a': self.sys.Line.rate_a.v[i] if hasattr(self.sys.Line, 'rate_a') else 0,
            })
        return lines
    
    def _get_load_data(self) -> List[Dict]:
        """Get all load data."""
        loads = []
        if hasattr(self.sys, 'PQ'):
            for i, idx in enumerate(self.sys.PQ.idx.v):
                loads.append({
                    'idx': idx,
                    'bus': self.sys.PQ.bus.v[i],
                    'name': self.sys.PQ.name.v[i] if hasattr(self.sys.PQ, 'name') else f"Load_{idx}",
                    'p0': self.sys.PQ.p0.v[i],
                    'q0': self.sys.PQ.q0.v[i],
                    'vn': self.sys.PQ.Vn.v[i] if hasattr(self.sys.PQ, 'Vn') else 1.0,
                })
        return loads
    
    def get_summary(self) -> Dict[str, Any]:
        """Get system summary statistics."""
        summary = {
            'case': self.case_path,
            'num_buses': len(self.sys.Bus.idx.v),
            'num_generators': len(self._get_generator_data()),
            'num_lines': len(self.sys.Line.idx.v),
            'num_loads': len(self._get_load_data()),
            'base_mva': self.sys.config.mva,
            'frequency': self.sys.config.freq,
        }
        return summary
    
    def print_summary(self):
        """Print formatted system summary."""
        summary = self.get_summary()
        print("\n" + "="*50)
        print("POWER SYSTEM SUMMARY")
        print("="*50)
        for key, value in summary.items():
            print(f"{key:20s}: {value}")
        print("="*50 + "\n")
    
    # ========================== NETWORK MODIFICATION ==========================
    
    def add_bus(self, idx: int, name: str = None, Vn: float = 110.0, 
                v0: float = 1.0, a0: float = 0.0, **kwargs) -> bool:
        """
        Add a new bus to the system.
        
        Args:
            idx: Bus index/ID
            name: Bus name
            Vn: Nominal voltage (kV)
            v0: Initial voltage magnitude (p.u.)
            a0: Initial voltage angle (rad)
        """
        try:
            self.sys.Bus.add(
                idx=idx,
                name=name or f"Bus_{idx}",
                Vn=Vn,
                v0=v0,
                a0=a0,
                **kwargs
            )
            self._build_graph()
            print(f"✓ Added bus {idx}: {name or f'Bus_{idx}'}")
            return True
        except Exception as e:
            print(f"✗ Error adding bus: {e}")
            return False
    
    def remove_bus(self, idx: int) -> bool:
        """Remove a bus from the system (and connected elements)."""
        try:
            # First remove connected lines
            lines_to_remove = []
            for i, line_idx in enumerate(self.sys.Line.idx.v):
                if self.sys.Line.bus1.v[i] == idx or self.sys.Line.bus2.v[i] == idx:
                    lines_to_remove.append(line_idx)
            
            for line_idx in lines_to_remove:
                self.remove_line(line_idx)
            
            # Remove generators at this bus
            for model_name in ['PV', 'Slack', 'GENROU', 'GENCLS']:
                if hasattr(self.sys, model_name):
                    model = getattr(self.sys, model_name)
                    gens_to_remove = []
                    for i, gen_idx in enumerate(model.idx.v):
                        if model.bus.v[i] == idx:
                            gens_to_remove.append(gen_idx)
                    for gen_idx in gens_to_remove:
                        self.remove_generator(gen_idx, model_name)
            
            # Remove loads at this bus
            if hasattr(self.sys, 'PQ'):
                loads_to_remove = []
                for i, load_idx in enumerate(self.sys.PQ.idx.v):
                    if self.sys.PQ.bus.v[i] == idx:
                        loads_to_remove.append(load_idx)
                for load_idx in loads_to_remove:
                    self.remove_load(load_idx)
            
            # Finally remove the bus
            bus_pos = list(self.sys.Bus.idx.v).index(idx)
            self.sys.Bus.idx.v = np.delete(self.sys.Bus.idx.v, bus_pos)
            
            self._build_graph()
            print(f"✓ Removed bus {idx}")
            return True
        except Exception as e:
            print(f"✗ Error removing bus: {e}")
            return False
    
    def add_line(self, idx: int, bus1: int, bus2: int, r: float, x: float,
                 b: float = 0.0, rate_a: float = 0.0, name: str = None, **kwargs) -> bool:
        """
        Add a transmission line between two buses.
        
        Args:
            idx: Line index
            bus1: From bus
            bus2: To bus
            r: Resistance (p.u.)
            x: Reactance (p.u.)
            b: Susceptance (p.u.)
            rate_a: Rating (MVA)
        """
        try:
            self.sys.Line.add(
                idx=idx,
                bus1=bus1,
                bus2=bus2,
                r=r,
                x=x,
                b=b,
                rate_a=rate_a,
                name=name or f"Line_{idx}",
                **kwargs
            )
            self._build_graph()
            print(f"✓ Added line {idx}: {bus1} -> {bus2}")
            return True
        except Exception as e:
            print(f"✗ Error adding line: {e}")
            return False
    
    def remove_line(self, idx: int) -> bool:
        """Remove a transmission line."""
        try:
            line_pos = list(self.sys.Line.idx.v).index(idx)
            self.sys.Line.idx.v = np.delete(self.sys.Line.idx.v, line_pos)
            self._build_graph()
            print(f"✓ Removed line {idx}")
            return True
        except Exception as e:
            print(f"✗ Error removing line: {e}")
            return False
    
    def add_generator(self, idx: int, bus: int, gen_type: str = 'PV',
                     Sn: float = 100.0, p0: float = 0.0, v0: float = 1.0,
                     name: str = None, **kwargs) -> bool:
        """
        Add a generator to a bus.
        
        Args:
            idx: Generator index
            bus: Bus index where generator is connected
            gen_type: Generator type ('PV', 'Slack', 'GENROU', etc.)
            Sn: Rated power (MVA)
            p0: Active power (p.u.)
            v0: Voltage setpoint (p.u.)
        """
        try:
            if not hasattr(self.sys, gen_type):
                print(f"✗ Generator type '{gen_type}' not available")
                return False
            
            model = getattr(self.sys, gen_type)
            model.add(
                idx=idx,
                bus=bus,
                Sn=Sn,
                p0=p0,
                v0=v0,
                name=name or f"{gen_type}_{idx}",
                **kwargs
            )
            print(f"✓ Added {gen_type} generator {idx} at bus {bus}")
            return True
        except Exception as e:
            print(f"✗ Error adding generator: {e}")
            return False
    
    def remove_generator(self, idx: int, gen_type: str = 'PV') -> bool:
        """Remove a generator."""
        try:
            if not hasattr(self.sys, gen_type):
                print(f"✗ Generator type '{gen_type}' not available")
                return False
            
            model = getattr(self.sys, gen_type)
            gen_pos = list(model.idx.v).index(idx)
            model.idx.v = np.delete(model.idx.v, gen_pos)
            print(f"✓ Removed {gen_type} generator {idx}")
            return True
        except Exception as e:
            print(f"✗ Error removing generator: {e}")
            return False
    
    def add_load(self, idx: int, bus: int, p0: float, q0: float,
                 name: str = None, **kwargs) -> bool:
        """Add a load to a bus."""
        try:
            self.sys.PQ.add(
                idx=idx,
                bus=bus,
                p0=p0,
                q0=q0,
                name=name or f"Load_{idx}",
                **kwargs
            )
            print(f"✓ Added load {idx} at bus {bus}")
            return True
        except Exception as e:
            print(f"✗ Error adding load: {e}")
            return False
    
    def remove_load(self, idx: int) -> bool:
        """Remove a load."""
        try:
            load_pos = list(self.sys.PQ.idx.v).index(idx)
            self.sys.PQ.idx.v = np.delete(self.sys.PQ.idx.v, load_pos)
            print(f"✓ Removed load {idx}")
            return True
        except Exception as e:
            print(f"✗ Error removing load: {e}")
            return False
    
    def modify_bus(self, idx: int, **kwargs) -> bool:
        """Modify bus parameters."""
        try:
            bus_pos = list(self.sys.Bus.idx.v).index(idx)
            for key, value in kwargs.items():
                if hasattr(self.sys.Bus, key):
                    getattr(self.sys.Bus, key).v[bus_pos] = value
            print(f"✓ Modified bus {idx}")
            return True
        except Exception as e:
            print(f"✗ Error modifying bus: {e}")
            return False

    # ========================== GRAPH & VISUALIZATION ==========================
    
    def _build_graph(self):
        """Build NetworkX graph from system data."""
        self.graph = nx.Graph()
        
        # Add nodes (buses)
        for bus_data in self._get_bus_data():
            self.graph.add_node(bus_data['idx'], **bus_data)
        
        # Add edges (lines)
        for line_data in self._get_line_data():
            self.graph.add_edge(
                line_data['bus1'],
                line_data['bus2'],
                **line_data
            )
        
        # Calculate layout
        self._calculate_positions()
    
    def _calculate_positions(self, layout: str = 'spring'):
        """Calculate node positions for visualization."""
        if self.graph is None:
            return
        
        if layout == 'spring':
            self.positions = nx.spring_layout(self.graph, seed=42, k=2, iterations=50)
        elif layout == 'circular':
            self.positions = nx.circular_layout(self.graph)
        elif layout == 'kamada':
            self.positions = nx.kamada_kawai_layout(self.graph)
        else:
            self.positions = nx.spring_layout(self.graph)
    
    def plot_network(self, figsize=(14, 10), layout='spring', show_labels=True,
                    show_generators=True, show_loads=True, save_path=None):
        """
        Plot the power system network with different shapes for components.
        
        Args:
            figsize: Figure size
            layout: Layout algorithm ('spring', 'circular', 'kamada')
            show_labels: Show bus labels
            show_generators: Show generator symbols
            show_loads: Show load symbols
            save_path: Path to save figure
        """

        if self.graph is None or self.graph.number_of_nodes() == 0:
            print("✗ No network loaded (graph is empty)")
            return

        self._calculate_positions(layout)
        if not self.positions:
            print("✗ No node positions computed")
            return

        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Get component data
        generators = self._get_generator_data()
        loads = self._get_load_data()
        
        # Create bus-to-component mapping
        gen_buses = {g['bus']: g for g in generators}
        load_buses = {l['bus']: l for l in loads}
        
        # Draw transmission lines first
        for line_data in self._get_line_data():
            bus1, bus2 = line_data['bus1'], line_data['bus2']
            if bus1 in self.positions and bus2 in self.positions:
                x1, y1 = self.positions[bus1]
                x2, y2 = self.positions[bus2]
                ax.plot([x1, x2], [y1, y2], 'gray', linewidth=2, alpha=0.6, zorder=1)
        
        # Draw buses
        for bus_idx, (x, y) in self.positions.items():
            # Bus circle
            circle = Circle((x, y), 0.03, color='lightblue', ec='black', 
                          linewidth=2, zorder=3)
            ax.add_patch(circle)
            
            # Bus label
            if show_labels:
                ax.text(x, y-0.08, str(bus_idx), ha='center', fontsize=9,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        # Draw generators
        if show_generators:
            for bus_idx, gen_data in gen_buses.items():
                if bus_idx in self.positions:
                    x, y = self.positions[bus_idx]
                    # Generator symbol (circle with G)
                    gen_circle = Circle((x+0.08, y+0.08), 0.04, color='green',
                                      ec='darkgreen', linewidth=2, zorder=4)
                    ax.add_patch(gen_circle)
                    ax.text(x+0.08, y+0.08, 'G', ha='center', va='center',
                           fontsize=10, fontweight='bold', color='white', zorder=5)
        
        # Draw loads
        if show_loads:
            for bus_idx, load_data in load_buses.items():
                if bus_idx in self.positions:
                    x, y = self.positions[bus_idx]
                    # Load symbol (triangle pointing down)
                    triangle = Polygon([(x-0.08, y-0.06), (x-0.08+0.05, y-0.06), 
                                      (x-0.08+0.025, y-0.12)],
                                     color='red', ec='darkred', linewidth=2, zorder=4)
                    ax.add_patch(triangle)
        
        # Legend
        legend_elements = [
            mpatches.Patch(color='lightblue', ec='black', label='Bus'),
            mpatches.Patch(color='gray', label='Transmission Line'),
        ]
        if show_generators:
            legend_elements.append(mpatches.Patch(color='green', label='Generator'))
        if show_loads:
            legend_elements.append(mpatches.Patch(color='red', label='Load'))
        
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
        
        ax.set_title(f"Power System Network: {os.path.basename(self.case_path) if self.case_path else 'Custom'}",
                    fontsize=14, fontweight='bold')
        ax.axis('off')
        ax.set_aspect('equal')
        
        plt.tight_layout()
        
        # TODO: check if folder exists

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Network plot saved to: {save_path}")
        
        # plt.show()

        plt.close()
    
    def plot_voltage_profile(self, figsize=(12, 6), save_path=None):
        """Plot voltage magnitude profile across all buses."""
        buses = self._get_bus_data()
        bus_ids = [b['idx'] for b in buses]
        voltages = [b['v0'] for b in buses]
        
        fig, ax = plt.subplots(figsize=figsize)
        ax.bar(range(len(bus_ids)), voltages, color='steelblue', edgecolor='black')
        ax.axhline(y=1.0, color='green', linestyle='--', linewidth=2, label='Nominal (1.0 p.u.)')
        ax.axhline(y=0.95, color='orange', linestyle='--', linewidth=1, label='Lower limit (0.95 p.u.)')
        ax.axhline(y=1.05, color='orange', linestyle='--', linewidth=1, label='Upper limit (1.05 p.u.)')
        
        ax.set_xlabel('Bus Index', fontsize=12)
        ax.set_ylabel('Voltage Magnitude (p.u.)', fontsize=12)
        ax.set_title('Bus Voltage Profile', fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(bus_ids)))
        ax.set_xticklabels(bus_ids, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        # plt.show()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Voltage profiles saved to: {save_path}")

        plt.close()

    def export_to_dataframe(self) -> Dict[str, pd.DataFrame]:
        """Export all system data as pandas DataFrames."""
        return {
            'buses': pd.DataFrame(self._get_bus_data()),
            'generators': pd.DataFrame(self._get_generator_data()),
            'lines': pd.DataFrame(self._get_line_data()),
            'loads': pd.DataFrame(self._get_load_data()),
        }

     # ========================== FREQUENCY & ROCOF METERS ==========================

    # ========================== FREQUENCY & ROCOF METERS ==========================

    def run_power_flow(self) -> bool:
        """Run power flow; returns True if converged."""
        try:
            return bool(self.sys.PFlow.run())
        except Exception as e:
            print(f"✗ PF error: {e}")
            return False

    def run_tds(self, tf: float = 10.0, criteria: int = 0) -> bool:
        """Run time-domain simulation to tf seconds."""
        try:
            self.sys.TDS.config.tf = float(tf)
            self.sys.TDS.config.criteria = int(criteria)
            return bool(self.sys.TDS.run())
        except Exception as e:
            print(f"✗ TDS error: {e}")
            return False

    def add_generator_trip(self, dev: str = "GENROU_2", t: float = 1.0):
        """
        Schedule a generator trip using an ANDES Toggle before setup().
        dev must match the SynGen device name, e.g. 'GENROU_2'.
        """
        try:
            self.sys.add(
                "Toggle",
                dict(model="SynGen", dev=dev, t=float(t))
            )
            print(f"✓ Trip scheduled: {dev} at t={t} s")
            return True
        except Exception as e:
            print(f"✗ Could not add trip: {e}")
            return False

    def _find_latest_out_files(self) -> Tuple[str, str]:
        """
        Find the most recent ANDES TDS output pair (<case>_out.lst / <case>_out.npz)
        in the current working directory.

        Returns
        -------
        (lst_path, npz_path)
        """
        import glob
        outs = glob.glob("*_out.npz")
        if not outs:
            raise RuntimeError("No '*_out.npz' file found in current directory. "
                            "Run TDS before calling frequency/ROCOF extraction.")
        # pick the most recently modified npz
        npz_path = max(outs, key=os.path.getmtime)

        # try to guess the matching .lst
        base = npz_path[:-4]  # strip ".npz"
        lst_guess = base + ".lst"
        if not os.path.exists(lst_guess):
            raise RuntimeError(f"Found {npz_path} but not the matching {lst_guess}. "
                            "Make sure you didn't move/rename files.")
        return (lst_guess, npz_path)

    def _parse_lst_headers(self, lst_path: str) -> Dict[int, str]:
        """
        Parse the ANDES <case>_out.lst file and return {col_index: var_name}.

        The .lst format per manual looks like:
            0, Time [s], Time [s]
            5, omega GENROU 1, $\\omega$ GENROU 1
            6, omega GENROU 2, $\\omega$ GENROU 2
        where first entry is the numeric column index in the NPZ `data`.

        Returns
        -------
        headers : dict[int,str]
            maps column index -> plain variable name (2nd column in .lst)
        """
        headers = {}
        with open(lst_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # split on comma only for first two commas
                # expected: idx, var_name, tex_name
                parts = [p.strip() for p in line.split(",", 2)]
                if len(parts) < 2:
                    continue
                try:
                    col_idx = int(parts[0])
                except ValueError:
                    continue
                var_name = parts[1]
                headers[col_idx] = var_name
        if not headers:
            raise RuntimeError(f"Could not parse any headers from {lst_path}")
        return headers

    def _extract_time_and_omega(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return (T_seconds, OmegaMat), using the official ANDES output files
        instead of guessing internal objects.

        T_seconds : shape [n_time,]
        OmegaMat  : shape [n_gen, n_time] (each row = one machine's ω(t))

        Steps:
        1. find the newest *_out.npz and its *_out.lst
        2. read *_out.lst to know which columns are time and ω
        3. load *_out.npz['data'] (n_time x n_vars)
        4. slice the right columns
        """
        import numpy as _np

        lst_path, npz_path = self._find_latest_out_files()

        # 1) parse headers from .lst
        headers = self._parse_lst_headers(lst_path)
        # headers looks like {0: 'Time [s]', 5: 'omega GENROU 1', ...}

        # 2) identify time column and omega columns
        time_col = None
        omega_cols = []
        for col_idx, name in headers.items():
            low = name.lower()
            if "time" in low and time_col is None:
                time_col = col_idx
            if "omega" in low and "genrou" in low:
                omega_cols.append(col_idx)

        if time_col is None:
            raise RuntimeError(
                f"Didn't find a 'Time' column in {lst_path}. "
                f"Available headers: {headers}"
            )
        if not omega_cols:
            # fallback: any 'omega'
            for col_idx, name in headers.items():
                if "omega" in name.lower():
                    omega_cols.append(col_idx)
        if not omega_cols:
            raise RuntimeError(
                f"Didn't find any generator speed ('omega') columns in {lst_path}. "
                f"Available headers: {headers}"
            )

        # sort omega columns in ascending order so GENROU_1,2,3,... stay ordered
        omega_cols = sorted(omega_cols)

        # 3) read npz
        z = _np.load(npz_path, allow_pickle=True)

        if "data" not in z.files:
            raise RuntimeError(
                f"{npz_path} does not contain 'data'. Keys = {list(z.files)}"
            )

        data = z["data"]
        # We expect data shape == (n_time, n_vars)
        if data.ndim != 2:
            raise RuntimeError(
                f"'data' in {npz_path} is not 2D. Shape = {data.shape}"
            )
        n_time, n_vars = data.shape
        # sanity: max col index must be < n_vars
        if max([time_col] + omega_cols) >= n_vars:
            raise RuntimeError(
                f"Column index > data.shape[1]. time_col={time_col}, "
                f"omega_cols={omega_cols}, n_vars={n_vars}"
            )

        # 4) slice arrays
        T = data[:, time_col].astype(float)             # shape [n_time]
        Omega_stack = data[:, omega_cols].T.astype(float)  # shape [n_gen, n_time]

        return T, Omega_stack

    def get_frequency_series(self, f_nom: float = 60.0, mode: str = "avg"
                             ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute frequency [Hz] and optionally ROCOF [Hz/s] from omega(t).

        mode:
          "avg" -> return the average freq across all machines.
          "all" -> return all machines separately (matrix).

        Returns
        -------
        T : np.ndarray, shape [n_time]
            time in seconds
        f : np.ndarray
            if mode == 'avg': shape [n_time]
            if mode == 'all': shape [n_gen, n_time]
        """
        import numpy as _np

        T, Omega = self._extract_time_and_omega()  # Omega: [n_gen, n_time]

        # convert per-unit speed deviation (?) to Hz.
        # In ANDES, omega for GENROU is rotor speed in p.u. of synchronous speed.
        # So electrical frequency in Hz is simply omega * f_nom.
        f_mat = Omega * f_nom  # [n_gen, n_time]

        if mode == "all":
            return T, f_mat
        elif mode == "avg":
            f_avg = _np.mean(f_mat, axis=0)  # [n_time]
            return T, f_avg
        else:
            raise ValueError("mode must be 'avg' or 'all'")

    def get_rocof_series(self, f_nom: float = 60.0, mode: str = "avg"
                        ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute ROCOF = df/dt [Hz/s].

        mode is same as get_frequency_series().
        """
        import numpy as _np

        T, f = self.get_frequency_series(f_nom=f_nom, mode=mode)

        # central difference-ish derivative
        if mode == "all":
            # f shape [n_gen, n_time]
            dfdt = _np.gradient(f, T, axis=1)
        else:
            # f shape [n_time]
            dfdt = _np.gradient(f, T)

        return T, dfdt

    def freq_metrics(self, T: np.ndarray, f_hz: np.ndarray) -> Dict[str, float]:
        """
        Compute frequency stability metrics for one or multiple frequency traces.

        Parameters
        ----------
        T : np.ndarray
            Time vector [s].
        f_hz : np.ndarray
            Frequency [Hz] (shape [n_time] or [n_gen, n_time]).

        Returns
        -------
        Dict[str, float]
            - f_nom        : nominal frequency [Hz]
            - f_nadir      : lowest frequency [Hz]
            - t_nadir      : time of lowest frequency [s]
            - rocof_min    : minimum RoCoF [Hz/s]
            - t_rocof_min  : time of min RoCoF [s]
            - rocof_max    : maximum RoCoF [Hz/s]
            - t_rocof_max  : time of max RoCoF [s]
            - overshoot    : peak frequency above nominal [Hz]
        """
        import numpy as np

        f_nom = float(getattr(self.sys.config, "freq", 50.0))

        # Average across generators if 2D
        if f_hz.ndim == 2:
            f_mean = np.mean(f_hz, axis=0)
        else:
            f_mean = f_hz

        # Compute RoCoF using numerical derivative
        rocof = np.gradient(f_mean, T)

        # --- Metrics ---
        i_nadir = int(np.argmin(f_mean))
        f_nadir = float(f_mean[i_nadir])
        t_nadir = float(T[i_nadir])

        i_rocof_min = int(np.argmin(rocof))
        i_rocof_max = int(np.argmax(rocof))

        overshoot = max(0.0, float(np.max(f_mean) - f_nom))

        return {
            "f_nom": f_nom,
            "f_nadir": f_nadir,
            "t_nadir": t_nadir,
            "rocof_min": float(rocof[i_rocof_min]),
            "t_rocof_min": float(T[i_rocof_min]),
            "rocof_max": float(rocof[i_rocof_max]),
            "t_rocof_max": float(T[i_rocof_max]),
            "overshoot": overshoot,
        }

    def plot_freq_and_rocof(
        self,
        T: np.ndarray,
        f_hz: np.ndarray,
        save_path: Optional[str] = None,
        title: str = "",
        show_nominal: bool = True,
    ):
        """
        Plot frequency (Hz) and RoCoF (Hz/s) over time.

        If f_hz is 2D (n_gen, n_time), plots the average in bold and
        individual traces in light lines.
        """
        import numpy as np
        import matplotlib.pyplot as plt

        f_nom = float(getattr(self.sys.config, "freq", 50.0))

        # Handle multiple generators
        if f_hz.ndim == 2:
            f_avg = np.mean(f_hz, axis=0)
            rocof_avg = np.gradient(f_avg, T)
            rocof_all = np.gradient(f_hz, T, axis=1)
        else:
            f_avg = f_hz
            rocof_avg = np.gradient(f_hz, T)
            rocof_all = None

        # --- Figure setup ---
        fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

        # --- Frequency subplot ---
        if f_hz.ndim == 2:
            axs[0].plot(T, f_hz.T, color="gray", alpha=0.4, lw=0.8)
        axs[0].plot(T, f_avg, color="blue", lw=2.2, label="Average Frequency")

        if show_nominal:
            axs[0].axhline(f_nom, color="k", ls="--", alpha=0.5, label=f"Nominal {f_nom:.1f} Hz")
        axs[0].set_ylabel("Frequency [Hz]", fontsize=11)
        axs[0].grid(alpha=0.3)
        axs[0].legend(loc="best", fontsize=9)
        axs[0].set_title("Frequency and RoCoF Evolution" if not title else title, fontsize=12, pad=8)

        # --- RoCoF subplot ---
        if rocof_all is not None:
            axs[1].plot(T, rocof_all.T, color="gray", alpha=0.4, lw=0.8)
        axs[1].plot(T, rocof_avg, color="orange", lw=2.2, label="Average RoCoF")
        axs[1].axhline(0, color="k", ls="--", alpha=0.4)
        axs[1].set_xlabel("Time [s]", fontsize=11)
        axs[1].set_ylabel("RoCoF [Hz/s]", fontsize=11)
        axs[1].grid(alpha=0.3)
        axs[1].legend(loc="best", fontsize=9)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"✓ Saved frequency/RoCoF plot to {save_path}")
        plt.close(fig)

# ========================== EXAMPLE USAGE ==========================

if __name__ == "__main__":
    # Create power system instance
    andes.config_logger(stream_level=20)

    case_path = "ieee14/ieee14_full.xlsx"

    ps = PowerSystem(case_path)
    
    # test_basic_features(ps)

    # test_pfl(ps)

    # test_with_real_vi(ps)

    ps.plot_network(save_path="network/vis_network.png")
    ps.plot_voltage_profile(save_path="network/voltage_profiles.png")
    # ps.plot_freq_and_rocof()
    l = ps._get_generator_data()
    for x in l:
        print(x)
    
    ps.add_generator(idx="GENREG_6", bus=12, gen_type="REGCA1", name="GENREG_6", gen=10)

    l = ps._get_generator_data()
    for x in l:
        print(x)

    ps.plot_network(save_path="network/vis_network_REG.png")