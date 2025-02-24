import numpy as np
import tempfile
import os
from skimage.draw import line_aa
import math
from ezdxf import recover
import streamlit as st

def process_dxf(file_data, size=10):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.dxf') as tmp_file:
            tmp_file.write(file_data.getvalue())
            tmp_file_path = tmp_file.name
        
        doc, auditor = recover.readfile(tmp_file_path)
        msp = doc.modelspace()
        
        lines = []
        x_coords = []
        y_coords = []
        
        # Process entities
        for entity in msp:
            if entity.dxftype() == 'LINE':
                start = entity.dxf.start
                end = entity.dxf.end
                lines.append([(float(start[0]), float(start[1])), (float(end[0]), float(end[1]))])
                x_coords.extend([float(start[0]), float(end[0])])
                y_coords.extend([float(start[1]), float(end[1])])
            
            elif entity.dxftype() == 'ARC':
                center = entity.dxf.center
                radius = float(entity.dxf.radius)
                start_angle = float(entity.dxf.start_angle)
                end_angle = float(entity.dxf.end_angle)
                
                start_angle = math.radians(start_angle)
                end_angle = math.radians(end_angle)
                if end_angle < start_angle:
                    end_angle += 2 * math.pi
                
                num_segments = 32
                angles = np.linspace(start_angle, end_angle, num_segments)
                
                for i in range(len(angles)-1):
                    x1 = center[0] + radius * math.cos(angles[i])
                    y1 = center[1] + radius * math.sin(angles[i])
                    x2 = center[0] + radius * math.cos(angles[i+1])
                    y2 = center[1] + radius * math.sin(angles[i+1])
                    lines.append([(x1, y1), (x2, y2)])
                    x_coords.extend([x1, x2])
                    y_coords.extend([y1, y2])
        
        # Process image
        return _process_lines_to_image(lines, x_coords, y_coords, size)
        
    except Exception as e:
        st.error(f"Error processing DXF: {str(e)}")
        return None

def _process_lines_to_image(lines, x_coords, y_coords, size):
    # Calculate bounds
    xmin, xmax = min(x_coords), max(x_coords)
    ymin, ymax = min(y_coords), max(y_coords)
    padding = 0.05 * max(xmax - xmin, ymax - ymin)
    
    xmin -= padding
    xmax += padding
    ymin -= padding
    ymax += padding
    
    # Create output image
    temp_size = 320
    high_res = np.zeros((temp_size, temp_size))
    
    # Draw lines
    for line in lines:
        x1 = int((line[0][0] - xmin) * (temp_size - 1) / (xmax - xmin))
        y1 = int((line[0][1] - ymin) * (temp_size - 1) / (ymax - ymin))
        x2 = int((line[1][0] - xmin) * (temp_size - 1) / (xmax - xmin))
        y2 = int((line[1][1] - ymin) * (temp_size - 1) / (ymax - ymin))
        
        rr, cc, val = line_aa(y1, x1, y2, x2)
        mask = (rr >= 0) & (rr < temp_size) & (cc >= 0) & (cc < temp_size)
        high_res[rr[mask], cc[mask]] = val[mask]
    
    # Convert to binary image
    outline = high_res > 0.1
    
    # Create output image
    img = np.zeros((size, size), dtype=np.float32)
    
    # Scale down the image
    scale_y = size / temp_size
    scale_x = size / temp_size
    
    for i in range(size):
        for j in range(size):
            y_start = int(i / scale_y)
            y_end = int((i + 1) / scale_y)
            x_start = int(j / scale_x)
            x_end = int((j + 1) / scale_x)
            
            block = outline[y_start:y_end+1, x_start:x_end+1]
            if np.any(block):
                img[i, j] = 1
    
    return 1 - img