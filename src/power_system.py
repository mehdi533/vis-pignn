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
    
    def load_case(self, case_path: str):
        """Load a power system case from file."""
        try:
            self.sys.reload(case_path)
            self.case_path = case_path
            self._build_graph()
            print(f"✓ Successfully loaded case: {case_path}")
            print(f"  - Buses: {len(self.sys.Bus.idx.v)}")
            print(f"  - Generators: {len(self.sys.PV.idx.v) if hasattr(self.sys, 'PV') else 0}")
            print(f"  - Lines: {len(self.sys.Line.idx.v)}")
            print(f"  - Loads: {len(self.sys.PQ.idx.v) if hasattr(self.sys, 'PQ') else 0}")
            return True
        except Exception as e:
            print(f"✗ Error loading case: {e}")
            return False
    import pandas as pd

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

    def save_case(self, output_path: str, format: str = 'xlsx'):
        """
        Save the current power system to file.
        
        Args:
            output_path: Output file path
            format: File format ('xlsx', 'json', 'csv')
        """
        try:
            if format == 'xlsx':
                xlsxwriter.write(self.sys, output_path)
            elif format == 'json':
                self._save_json(output_path)
            else:
                print(f"Format '{format}' not supported. Use 'xlsx' or 'json'")
                return False
            
            print(f"✓ Saved case to: {output_path}")
            return True
        except Exception as e:
            print(f"✗ Error saving case: {e}")
            return False
    
    def _save_json(self, output_path: str):
        """Save system data as JSON."""
        data = {
            'buses': self._get_bus_data(),
            'generators': self._get_generator_data(),
            'lines': self._get_line_data(),
            'loads': self._get_load_data(),
        }
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
    
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
        gen_models = ['PV', 'Slack', 'GENROU', 'GENCLS']
        
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
                        'p0': model.p0.v[i] if hasattr(model, 'p0') else 0,
                        'v0': model.v0.v[i] if hasattr(model, 'v0') else 1.0,
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
    
    def plot_voltage_profile(self, figsize=(12, 6)):
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
        
        # TODO: add save

        plt.close()
    
    def export_to_dataframe(self) -> Dict[str, pd.DataFrame]:
        """Export all system data as pandas DataFrames."""
        return {
            'buses': pd.DataFrame(self._get_bus_data()),
            'generators': pd.DataFrame(self._get_generator_data()),
            'lines': pd.DataFrame(self._get_line_data()),
            'loads': pd.DataFrame(self._get_load_data()),
        }


# ========================== EXAMPLE USAGE ==========================

if __name__ == "__main__":
    # Create power system instance
    andes.config_logger(stream_level=50)

    # wf = "venv/lib/python3.11/site-packages/andes/cases/"
    # case_path = "ieee14/ieee14_full.xlsx"
    # case_path = os.path.join(wf, case_path)

    ps = PowerSystem("modified_system.xlsx")
    
    # # Print summary

    # TODO: not sure about nb generators
    ps.print_summary()
    
    # # Get data as DataFrames
    dfs = ps.export_to_dataframe()
    print("\nBuses DataFrame:")
    print(dfs['buses'].head())
    
    # # Visualize network
    # TODO: why not same nb of generators, change PV to another
    ps.plot_network(layout='spring', show_labels=True, save_path="network/full_image.png")
    
    # # Plot voltage profile
    ps.plot_voltage_profile()
    
    # # Example: Add a new bus
    # TODO: when adding, check that the same bus doesn't already exist
    ps.add_bus(idx=999, name="New_Bus", Vn=230.0)
    
    # # Example: Add a line
    ps.add_line(idx=999, bus1=1, bus2=999, r=0.01, x=0.1)
    
    # # Example: Add a generator
    ps.add_generator(idx=999, bus=999, gen_type='PV', Sn=100, p0=0.5)
    
    # # Save modified system
    ps.plot_network(layout='spring', show_labels=True, save_path="network/modified_image.png")
    ps.save_case_xlsxwriter('modified_system.xlsx')