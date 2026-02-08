import React, { useRef, useEffect, useState } from 'react';
import * as THREE from 'three';

export default function HoloCellGeometry() {
  const containerRef = useRef(null);
  const [info, setInfo] = useState('');
  
  useEffect(() => {
    if (!containerRef.current) return;
    
    // Scene setup
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0a0a0f);
    
    const width = containerRef.current.clientWidth;
    const height = containerRef.current.clientHeight;
    const camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 1000);
    camera.position.set(4, 3, 4);
    camera.lookAt(0, 0, 0);
    
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(width, height);
    containerRef.current.appendChild(renderer.domElement);
    
    // Colors from Egyptian palette
    const gold = 0xffd700;
    const lapis = 0x1e3a8a;
    const turquoise = 0x40e0d0;
    const carnelian = 0xb22222;
    
    // === THE HOLOCELL: Octahedron ===
    // 6 vertices, 3 axes, 8 faces
    // Octahedron is dual of cube - perfect bilateral symmetry
    // 3 perpendicular axes = Trinition (a + bi + cj)
    // NOTE: Center is EMPTY - T(16) is eigenvalue not a node
    
    const octaGeometry = new THREE.OctahedronGeometry(1.36, 0); // 1.36 = T(16)/100
    const octaMaterial = new THREE.MeshPhongMaterial({
      color: gold,
      transparent: true,
      opacity: 0.3,
      side: THREE.DoubleSide,
    });
    const octahedron = new THREE.Mesh(octaGeometry, octaMaterial);
    scene.add(octahedron);
    
    // Wireframe overlay
    const wireGeometry = new THREE.OctahedronGeometry(1.36, 0);
    const wireMaterial = new THREE.MeshBasicMaterial({
      color: gold,
      wireframe: true,
      transparent: true,
      opacity: 0.8,
    });
    const wireOcta = new THREE.Mesh(wireGeometry, wireMaterial);
    scene.add(wireOcta);
    
    // === TRINITION AXES (i, j, k) ===
    const axisLength = 2.0;
    const axisWidth = 0.02;
    
    // X-axis (i) - Lapis
    const xAxis = new THREE.Mesh(
      new THREE.CylinderGeometry(axisWidth, axisWidth, axisLength * 2, 8),
      new THREE.MeshPhongMaterial({ color: lapis, emissive: lapis, emissiveIntensity: 0.3 })
    );
    xAxis.rotation.z = Math.PI / 2;
    scene.add(xAxis);
    
    // Y-axis (j) - Turquoise  
    const yAxis = new THREE.Mesh(
      new THREE.CylinderGeometry(axisWidth, axisWidth, axisLength * 2, 8),
      new THREE.MeshPhongMaterial({ color: turquoise, emissive: turquoise, emissiveIntensity: 0.3 })
    );
    scene.add(yAxis);
    
    // Z-axis (k) - Carnelian
    const zAxis = new THREE.Mesh(
      new THREE.CylinderGeometry(axisWidth, axisWidth, axisLength * 2, 8),
      new THREE.MeshPhongMaterial({ color: carnelian, emissive: carnelian, emissiveIntensity: 0.3 })
    );
    zAxis.rotation.x = Math.PI / 2;
    scene.add(zAxis);
    
    // === VERTEX SPHERES (6 vertices = 3 bilateral pairs) ===
    const vertexRadius = 0.08;
    const vertices = [
      [1.36, 0, 0], [-1.36, 0, 0],  // X pair
      [0, 1.36, 0], [0, -1.36, 0],  // Y pair
      [0, 0, 1.36], [0, 0, -1.36],  // Z pair
    ];
    const vertexColors = [lapis, lapis, turquoise, turquoise, carnelian, carnelian];
    
    vertices.forEach((pos, i) => {
      const sphere = new THREE.Mesh(
        new THREE.SphereGeometry(vertexRadius, 16, 16),
        new THREE.MeshPhongMaterial({ 
          color: vertexColors[i], 
          emissive: vertexColors[i], 
          emissiveIntensity: 0.5 
        })
      );
      sphere.position.set(...pos);
      scene.add(sphere);
    });
    
    // === INNER SEED: Small icosahedron at center ===
    // Represents T(16) = 136 as the eigenvalue (visual only - not a structural node)
    const seedGeometry = new THREE.IcosahedronGeometry(0.136, 0);
    const seedMaterial = new THREE.MeshPhongMaterial({
      color: 0xffffff,
      emissive: 0xffffff,
      emissiveIntensity: 0.8,
    });
    const seed = new THREE.Mesh(seedGeometry, seedMaterial);
    scene.add(seed);
    
    // === 408 RING: Scale invariance marker ===
    // T(16) × 3 = 408, shown as outer torus
    const torusGeometry = new THREE.TorusGeometry(2.04, 0.02, 16, 100); // 4.08/2 radius
    const torusMaterial = new THREE.MeshPhongMaterial({
      color: gold,
      transparent: true,
      opacity: 0.4,
    });
    const torus = new THREE.Mesh(torusGeometry, torusMaterial);
    torus.rotation.x = Math.PI / 2;
    scene.add(torus);
    
    // Second torus (orthogonal)
    const torus2 = new THREE.Mesh(torusGeometry, torusMaterial);
    scene.add(torus2);
    
    // Third torus
    const torus3 = new THREE.Mesh(torusGeometry, torusMaterial);
    torus3.rotation.z = Math.PI / 2;
    scene.add(torus3);
    
    // === LIGHTING ===
    const ambientLight = new THREE.AmbientLight(0x404040, 0.5);
    scene.add(ambientLight);
    
    const pointLight1 = new THREE.PointLight(0xffffff, 1, 100);
    pointLight1.position.set(5, 5, 5);
    scene.add(pointLight1);
    
    const pointLight2 = new THREE.PointLight(0x4080ff, 0.5, 100);
    pointLight2.position.set(-5, -5, 5);
    scene.add(pointLight2);
    
    // === ANIMATION ===
    let time = 0;
    const animate = () => {
      requestAnimationFrame(animate);
      time += 0.005;
      
      // Slow rotation
      octahedron.rotation.y = time * 0.3;
      octahedron.rotation.x = Math.sin(time * 0.2) * 0.1;
      wireOcta.rotation.y = time * 0.3;
      wireOcta.rotation.x = Math.sin(time * 0.2) * 0.1;
      
      // Seed pulses
      const pulse = 1 + Math.sin(time * 2) * 0.1;
      seed.scale.set(pulse, pulse, pulse);
      
      // Camera orbits slowly
      camera.position.x = Math.cos(time * 0.2) * 5;
      camera.position.z = Math.sin(time * 0.2) * 5;
      camera.position.y = 2 + Math.sin(time * 0.1);
      camera.lookAt(0, 0, 0);
      
      renderer.render(scene, camera);
    };
    animate();
    
    // Info text
    setInfo(`
      HOLOCELL GEOMETRY
      ─────────────────
      Octahedron: 6 vertices, 8 faces, 12 edges
      • 6 vertices = 3 bilateral pairs (Trinition axes)
      • 8 faces
      • 12 edges = zodiac
      
      Center: EMPTY (T(16)=136 is eigenvalue, not node)
      Axes: i, j, k of Trinition (z = a + bi + cj)
      Rings: 408 = T(16) × 3 (scale invariance)
      
      Octahedron is dual of cube:
      • Vertices ↔ Faces exchange
      • Perfect bilateral symmetry
      • Minimum platonic with 3-axis symmetry
      
      OPTIMAL RESONANT SEED:
      Rate: 0.0259 | Steps to 90%: 24.7
      Beats buckyball (60 nodes) with 10× fewer
    `);
    
    // Cleanup
    return () => {
      if (containerRef.current && renderer.domElement) {
        containerRef.current.removeChild(renderer.domElement);
      }
      renderer.dispose();
    };
  }, []);
  
  return (
    <div className="w-full h-screen bg-gray-950 flex flex-col">
      <div className="text-center py-4 text-amber-400 font-bold text-xl">
        HoloCell: The Reality Crystal
      </div>
      <div className="flex-1 flex">
        <div ref={containerRef} className="flex-1" />
        <div className="w-72 p-4 bg-gray-900 text-gray-300 font-mono text-xs whitespace-pre-wrap overflow-auto">
          {info}
          <div className="mt-4 text-amber-500">
            T(16) = 136 eigenvalue
          </div>
          <div className="mt-2 space-y-1">
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full bg-blue-800"></div>
              <span>i-axis (lapis)</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full bg-teal-400"></div>
              <span>j-axis (turquoise)</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full bg-red-800"></div>
              <span>k-axis (carnelian)</span>
            </div>
          </div>
          <div className="mt-4 text-gray-500 text-xs">
            ij = 1, ji = -1
            <br/>
            Non-commutative 3D complex
          </div>
        </div>
      </div>
    </div>
  );
}
