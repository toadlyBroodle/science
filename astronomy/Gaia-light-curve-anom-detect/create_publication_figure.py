#!/usr/bin/env python3
"""
Create publication-quality figure for TIC 22888126 showing all TESS sectors
without time gaps, suitable for RNAAS submission.

Run this after running TIC22888126_Complete_LightCurve.ipynb to generate
the sector_data pickle file, OR run standalone with hardcoded data paths.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pickle
import os

# Try to load from pickle if available, otherwise we'll need the notebook data
DATA_FILE = 'tic22888126_sector_data.pkl'

def create_concatenated_figure(sector_data):
    """Create figure with sectors plotted side-by-side without gaps."""
    
    fig, ax = plt.subplots(figsize=(14, 4))
    
    colors = {
        13: '#1f77b4',   # Blue
        39: '#d62728',   # Red  
        66: '#9467bd',   # Purple
        93: '#2ca02c',   # Green
    }
    
    # Plot each sector concatenated
    x_offset = 0
    tick_positions = []
    tick_labels = []
    sector_boundaries = [0]
    
    for sector in sorted(sector_data.keys()):
        sd = sector_data[sector]
        time = sd['time']
        flux = sd['flux']
        
        # Normalize time to start at 0 for this sector
        time_normalized = time - time.min()
        
        # Plot with offset
        ax.scatter(time_normalized + x_offset, flux, s=1, alpha=0.6, 
                   c=colors.get(sector, 'gray'), label=f'Sector {sector}')
        
        # Mark sector center for tick
        sector_center = x_offset + (time_normalized.max() - time_normalized.min()) / 2
        tick_positions.append(sector_center)
        tick_labels.append(f'S{sector}')
        
        # Update offset for next sector (add small gap for visual separation)
        x_offset += time_normalized.max() + 2
        sector_boundaries.append(x_offset - 1)
    
    # Add vertical lines between sectors
    for boundary in sector_boundaries[1:-1]:
        ax.axvline(boundary, color='gray', linestyle=':', alpha=0.5, lw=0.5)
    
    # Outburst threshold line
    ax.axhline(1.5, color='red', linestyle='--', alpha=0.5, lw=1)
    ax.axhline(1.0, color='gray', linestyle='-', alpha=0.3, lw=0.5)
    
    # Labels
    ax.set_xlabel('Days from sector start (concatenated)', fontsize=11)
    ax.set_ylabel('Normalized Flux', fontsize=11)
    ax.set_title('TIC 22888126: Complete TESS Light Curve (6 years, 4 sectors)\n'
                 'Seven outbursts across 4 sectors; candidate P ~ 90 min (period gap boundary)',
                 fontsize=12, fontweight='bold')
    
    # Custom x-ticks showing sector numbers
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, fontsize=10)
    
    # Add secondary x-axis showing actual observation span
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks([tick_positions[0], tick_positions[-1]])
    ax2.set_xticklabels(['2019 Jul', '2025 Jul'], fontsize=9)
    ax2.set_xlabel('Observation epoch', fontsize=9)
    
    # Legend
    ax.legend(loc='upper right', markerscale=5, fontsize=9)
    
    # Y-axis limit to show outburst clearly
    ax.set_ylim(-0.1, 13)
    
    plt.tight_layout()
    return fig


def create_panel_figure(sector_data):
    """Create 4-panel figure with each sector in its own subplot."""
    
    fig = plt.figure(figsize=(14, 8))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.25)
    
    sectors = sorted(sector_data.keys())
    positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
    
    colors = {
        13: '#1f77b4',
        39: '#d62728',
        66: '#9467bd', 
        93: '#2ca02c',
    }
    
    for (row, col), sector in zip(positions, sectors):
        ax = fig.add_subplot(gs[row, col])
        
        sd = sector_data[sector]
        time = sd['time']
        flux = sd['flux']
        
        # Normalize time
        time_norm = time - time.min()
        
        ax.scatter(time_norm, flux, s=2, alpha=0.6, c=colors[sector])
        
        # Mark outliers
        med = np.nanmedian(flux)
        std = np.nanstd(flux)
        outliers = flux > med + 3*std
        if np.any(outliers):
            ax.scatter(time_norm[outliers], flux[outliers], s=8, c='red', 
                       zorder=10, label=f'>3σ: {np.sum(outliers)}')
            ax.legend(loc='upper right', fontsize=8)
        
        ax.axhline(1.0, color='gray', linestyle='-', alpha=0.3)
        ax.set_xlabel('Days', fontsize=9)
        ax.set_ylabel('Normalized Flux', fontsize=9)
        
        # Title with sector info
        status = 'OUTBURST' if sector == 13 else 'Quiescent'
        ax.set_title(f'Sector {sector} ({status})', fontsize=10, fontweight='bold')
        
        # Same y-scale for comparison
        ax.set_ylim(-0.1, 13 if sector == 13 else 2)
        ax.grid(True, alpha=0.2)
    
    fig.suptitle('TIC 22888126 / Gaia DR3 5947829831449228800\n'
                 'Dwarf Nova Candidate | P ~ 90 min (TESS L-S) | '
                 '7 outbursts in 6 years',
                 fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    return fig


if __name__ == '__main__':
    # Check if we have saved data
    if os.path.exists(DATA_FILE):
        print(f"Loading data from {DATA_FILE}")
        with open(DATA_FILE, 'rb') as f:
            sector_data = pickle.load(f)
    else:
        print(f"Data file {DATA_FILE} not found.")
        print("Please run TIC22888126_Complete_LightCurve.ipynb first,")
        print("then save sector_data with:")
        print("  import pickle")
        print("  with open('tic22888126_sector_data.pkl', 'wb') as f:")
        print("      pickle.dump(sector_data, f)")
        exit(1)
    
    # Create concatenated figure
    print("\nCreating concatenated figure...")
    fig1 = create_concatenated_figure(sector_data)
    fig1.savefig('tic22888126_concatenated.png', dpi=200, bbox_inches='tight',
                 facecolor='white', edgecolor='none')
    print("Saved: tic22888126_concatenated.png")
    
    # Create panel figure
    print("\nCreating panel figure...")
    fig2 = create_panel_figure(sector_data)
    fig2.savefig('tic22888126_panels.png', dpi=200, bbox_inches='tight',
                 facecolor='white', edgecolor='none')
    print("Saved: tic22888126_panels.png")
    
    plt.show()
    print("\nDone!")
