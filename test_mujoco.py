#!/usr/bin/env python3
"""
Test script to verify MuJoCo 3.0.1 installation and functionality.
Run this script to ensure your MuJoCo environment is set up correctly.
"""

import mujoco
import numpy as np

def test_mujoco_basic():
    """Test basic MuJoCo functionality."""
    print("Testing MuJoCo basic functionality...")
    
    # Test 1: Import and version check
    print(f"✅ MuJoCo version: {mujoco.__version__}")
    
    # Test 2: Create a simple model
    xml_string = """
    <mujoco>
        <worldbody>
            <geom name="ground" type="plane" size="1 1 0.1" rgba="0.8 0.6 0.4 1"/>
            <body name="box" pos="0 0 0.5">
                <geom name="box_geom" type="box" size="0.1 0.1 0.1" rgba="0.2 0.4 0.8 1"/>
                <joint name="box_joint" type="free"/>
            </body>
        </worldbody>
    </mujoco>
    """
    
    try:
        model = mujoco.MjModel.from_xml_string(xml_string)
        data = mujoco.MjData(model)
        print("✅ Model creation successful")
        
        # Test 3: Run a simulation step
        mujoco.mj_step(model, data)
        print("✅ Simulation step successful")
        
        # Test 4: Check model properties
        print(f"✅ Model has {model.nq} position coordinates")
        print(f"✅ Model has {model.nv} velocity coordinates")
        print(f"✅ Model has {model.nbody} bodies")
        print(f"✅ Model has {model.ngeom} geometries")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_mujoco_rendering():
    """Test MuJoCo rendering capabilities."""
    print("\nTesting MuJoCo rendering...")
    
    try:
        # Try to create a renderer (this might fail in headless environments)
        xml_string = """
        <mujoco>
            <worldbody>
                <geom name="ground" type="plane" size="1 1 0.1"/>
            </worldbody>
        </mujoco>
        """
        
        model = mujoco.MjModel.from_xml_string(xml_string)
        data = mujoco.MjData(model)
        
        # Try to create a renderer
        renderer = mujoco.Renderer(model)
        print("✅ Renderer creation successful")
        
        # Try to render
        renderer.update_scene(data)
        pixels = renderer.render()
        print(f"✅ Rendering successful, image shape: {pixels.shape}")
        
        return True
        
    except Exception as e:
        print(f"⚠️  Rendering test failed (this is normal in headless environments): {e}")
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("MuJoCo 3.0.1 Test Suite")
    print("=" * 50)
    
    # Run basic tests
    basic_success = test_mujoco_basic()
    
    # Run rendering tests
    rendering_success = test_mujoco_rendering()
    
    print("\n" + "=" * 50)
    if basic_success:
        print("🎉 MuJoCo is working correctly!")
        print("You can now use 'import mujoco' in your code.")
    else:
        print("❌ MuJoCo setup has issues. Please check your environment.")
    print("=" * 50)
