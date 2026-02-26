def create_bloch_widget():

    """
        Creates and displays the interactive Bloch sphere widget.
        Must be called from a Jupyter notebook with %matplotlib widget enabled.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from ipywidgets import VBox, HBox, FloatSlider, HTML, interactive_output
    from mpl_toolkits.mplot3d import proj3d
    from IPython.display import display

    # Figure setup -------------------------------
    fig = plt.figure(figsize=(5,5))

    ax = fig.add_subplot(111, projection='3d')
    ax.set_box_aspect([1,1,1])
    ax.grid(False)
    ax.set_axis_off()
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    fig.canvas.header_visible = False
    fig.canvas.footer_visible = False

    # Draw Bloch sphere -------------------------------

    # Transparent but smooth Bloch sphere
    u, v = np.mgrid[0:2*np.pi:120j, 0:np.pi:60j]  # denser mesh for smoothness
    xs = np.cos(u) * np.sin(v)
    ys = np.sin(u) * np.sin(v)
    zs = np.cos(v)

    # Transparent surface with subtle shading
    ax.plot_surface(xs, ys, zs, color='skyblue', alpha=0.25, rstride=1, cstride=1, shade=True)

    # Planes y=0 and z=0
    px = np.linspace(-1.2,1.2,50)
    pz = np.linspace(-1.2,1.2,50)
    PX, PZ = np.meshgrid(px, pz)
    ax.plot_surface(PX, 0*PX, PZ, color='blue', alpha=0.03)

    px2 = np.linspace(-1.2,1.2,50)
    py2 = np.linspace(-1.2,1.2,50)
    PX2, PY2 = np.meshgrid(px2, py2)
    ax.plot_surface(PX2, PY2, 0*PX2, color='red', alpha=0.03)

    # Plane x=0 (YZ plane)
    py = np.linspace(-1.2, 1.2, 50)
    pz = np.linspace(-1.2, 1.2, 50)
    PY, PZ = np.meshgrid(py, pz)
    ax.plot_surface(0*PY, PY, PZ, color='green', alpha=0.03)  # green translucent plane

    # Axes
    ax.quiver(0,0,0,1.5,0,0,color='r', arrow_length_ratio=0.08)
    ax.quiver(0,0,0,0,1.5,0,color='g', arrow_length_ratio=0.08)
    ax.quiver(0,0,0,0,0,1.5,color='b', arrow_length_ratio=0.08)
    ax.text(1.25,0,0,'X', color='r', fontsize=12)
    ax.text(0,1.25,0,'Y', color='g', fontsize=12)
    ax.text(0.1,0,1.4,'Z', color='b', fontsize=12)

    # Quantum state labels that rotate in 3D -----------------------------------------

    # |0> at north pole
    ax.text(0, 0.05, 1.2, r"$|0\rangle$", color="k", fontsize=12,
            horizontalalignment='center', verticalalignment='center')

    # |1> at south pole
    ax.text(0, 0, -1.3, r"$|1\rangle$", color="k", fontsize=12,
            horizontalalignment='center', verticalalignment='center')

    
    # Initial angles -------------------------------
    theta0_deg = 0
    phi0_deg   = 0
    theta0 = np.deg2rad(theta0_deg)
    phi0   = np.deg2rad(phi0_deg)

    x = np.sin(theta0)*np.cos(phi0)
    y = np.sin(theta0)*np.sin(phi0)
    z = np.cos(theta0)

    # Store artists
    artists = {}
    artists['bloch_vec'] = ax.quiver(0,0,0,x,y,z,color='k', arrow_length_ratio=0.15)
    artists['phi_arc'], = ax.plot([], [], [], 'r', lw=2)
    artists['theta_arc'], = ax.plot([], [], [], 'b', lw=2)

    artists['phi_label'] = ax.text(0,0,0,'', color='darkred', fontsize=12)
    artists['theta_label'] = ax.text(0,0,0,'', color='darkblue', fontsize=12)

    artists['proj_xy'], = ax.plot([], [], [], 'k:', lw=2)   # dotted XY-plane projection
    artists['proj_z'],  = ax.plot([], [], [], 'k:', lw=2)  # Z-axis projection
    artists['origin_to_xy'], = ax.plot([], [], [], 'k:', lw=1.5)

    line_pts = np.linspace(-1.2, 1.2, 100)

    # Intersection of XY plane (z=0) and XZ plane (y=0) → x-axis
    ax.plot(line_pts, 0*line_pts, 0*line_pts, 'k--', lw=1)

    # Optional: intersection of XY and YZ planes → y-axis
    ax.plot(0*line_pts, line_pts, 0*line_pts, 'k--', lw=1)

    # Optional: intersection of XZ and YZ planes → z-axis
    ax.plot(0*line_pts, 0*line_pts, line_pts, 'k--', lw=1)

    # Intersection of planes with the sphere (dotted) -------------------------------
    phi_vals = np.linspace(0, 2*np.pi, 200)

    # Intersection with y=0 plane (XZ plane)
    x_y0 = np.cos(phi_vals)
    y_y0 = np.zeros_like(phi_vals)
    z_y0 = np.sin(phi_vals)
    ax.plot(x_y0, y_y0, z_y0, 'b--', lw=1)

    # Intersection with z=0 plane (XY plane)
    x_z0 = np.cos(phi_vals)
    y_z0 = np.sin(phi_vals)
    z_z0 = np.zeros_like(phi_vals)
    ax.plot(x_z0, y_z0, z_z0, 'r--', lw=1)

    # Intersection with x=0 plane (YZ plane) → cyan dotted
    x_x0 = np.zeros_like(phi_vals)
    y_x0 = np.cos(phi_vals)
    z_x0 = np.sin(phi_vals)
    ax.plot(x_x0, y_x0, z_x0, 'g--', lw=1)  # cyan dotted

    info_box = HTML()

    def add_2d_label(text, x, y, z, ax, dx=0.0, dy=0.0, size=14, color='k'):
        """Places screen-aligned text near a 3D point."""
        X, Y, _ = proj3d.proj_transform(x, y, z, ax.get_proj())
        ax.text2D(X + dx, Y + dy, text, fontsize=size, color=color)

    # Sliders -------------------------------
    theta_slider = FloatSlider(value=theta0_deg, min=0, max=180, step=1, description='θ°')
    phi_slider   = FloatSlider(value=phi0_deg, min=0, max=360, step=1, description='φ°')

    # Update function  -------------------------------
    def update(theta_deg, phi_deg, artists=artists):
        theta = np.deg2rad(theta_deg)
        phi = np.deg2rad(phi_deg)

        alpha = np.cos(theta/2)
        beta  = np.exp(1j*phi) * np.sin(theta/2)

        # Update Bloch vector
        artists['bloch_vec'].remove()
        x = np.sin(theta)*np.cos(phi)
        y = np.sin(theta)*np.sin(phi)
        z = np.cos(theta)
        artists['bloch_vec'] = ax.quiver(0,0,0,x,y,z,color='k', arrow_length_ratio=0.15)
    
    
    # Remove old surfaces if they exist
        if 'phi_surface' in artists:
            artists['phi_surface'].remove()     
        if 'theta_surface' in artists:
            artists['theta_surface'].remove()

        # Scale factor for arcs
        r = 0.4

        # φ arc (red)
        phi_vals = np.linspace(0, phi, 50)
        r_vals = np.linspace(0, r, 2)
        PHI, R = np.meshgrid(phi_vals, r_vals)
        X = R * np.cos(PHI) * np.sin(theta)
        Y = R * np.sin(PHI) * np.sin(theta)
        Z = R * np.zeros_like(PHI)
        artists['phi_surface'] = ax.plot_surface(X, Y, Z, color='red', alpha=0.3)
        
        # φ arc edge line
        X_edge = r * np.cos(phi_vals) * np.sin(theta)
        Y_edge = r * np.sin(phi_vals) * np.sin(theta)
        Z_edge = np.zeros_like(phi_vals)
        if 'phi_edge' in artists:
            artists['phi_edge'].remove()
        artists['phi_edge'], = ax.plot(X_edge, Y_edge, Z_edge, color='darkred', lw=2)

        
        # θ arc (blue)
        theta_vals = np.linspace(0, theta, 50)
        r_vals = np.linspace(0, r, 2)
        THETA, R = np.meshgrid(theta_vals, r_vals)
        VX = np.cos(phi)
        VY = np.sin(phi)
        X = R * np.sin(THETA) * VX
        Y = R * np.sin(THETA) * VY
        Z = R * np.cos(THETA)
        artists['theta_surface'] = ax.plot_surface(X, Y, Z, color='blue', alpha=0.3)

        # θ arc edge line
        X_edge = r * np.sin(theta_vals) * VX
        Y_edge = r * np.sin(theta_vals) * VY
        Z_edge = r * np.cos(theta_vals)
        if 'theta_edge' in artists:
            artists['theta_edge'].remove()
        artists['theta_edge'], = ax.plot(X_edge, Y_edge, Z_edge, color='darkblue', lw=2)

    
        # φ and θ labels
        if 'phi_label' in artists:
            artists['phi_label'].remove()
        if 'theta_label' in artists:
            artists['theta_label'].remove()

        phi_mid = phi / 2
        x_phi = r * np.cos(phi_mid) * np.sin(theta)
        y_phi = r * np.sin(phi_mid) * np.sin(theta)
        z_phi = 0

        theta_mid = theta / 2
        VX = np.cos(phi)
        VY = np.sin(phi)
        x_theta = r * np.sin(theta_mid) * VX
        y_theta = r * np.sin(theta_mid) * VY
        z_theta = r * np.cos(theta_mid)

        # Add screen-aligned labels
        artists['phi_label'] = ax.text2D(
            *proj3d.proj_transform(x_phi, y_phi, z_phi, ax.get_proj())[:2],
            "φ", fontsize=16, color='darkred'
        )

        artists['theta_label'] = ax.text2D(
            *proj3d.proj_transform(x_theta, y_theta, z_theta, ax.get_proj())[:2],
            "θ", fontsize=16, color='darkblue'
        )

        # --- Projection ONTO Z-axis (vertical) ---
        artists['proj_z'].set_data([0,x], [0,y])
        artists['proj_z'].set_3d_properties([z, z])  # dashed line to Z axis

        # --- Projection ONTO XY-plane (vector shadow) ---
        artists['proj_xy'].set_data([x, x], [y, y])  
        artists['proj_xy'].set_3d_properties([z, 0])   # dotted line down to plane
    
        # --- Line from origin (0,0,0) to XY-plane projection (x,y,0) ---
        artists['origin_to_xy'].set_data([0, x], [0, y])
        artists['origin_to_xy'].set_3d_properties([0, 0])


        # Update info
        info_box.value = (
            # f"<b>θ = {theta_deg:.1f}°</b> &nbsp;&nbsp;"
            # f"<b>φ = {phi_deg:.1f}°</b><br>"
            f"α =  {alpha:.3f}, &nbsp;&nbsp;&nbsp;"
            f"β = {beta.real:.3f} + {beta.imag:.3f} i"
        )
        # fig.canvas.draw_idle()
        fig.canvas.draw()

    # Interactive output -------------------------------

    out = interactive_output(update, {'theta_deg': theta_slider, 'phi_deg': phi_slider})

    # Layout: figure left, sliders right
    controls = VBox([theta_slider, phi_slider, info_box])
    display(HBox([controls]))

    # layout=Layout(align_items='center', justify_content='center')
    # Initialize
    update(theta0_deg, phi0_deg)

    return