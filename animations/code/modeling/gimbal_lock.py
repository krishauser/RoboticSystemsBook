from manimlib import *
import numpy as np
from klampt.math import so3

manim_config.camera.background_color = "#FFFFFF"  # Set the background color to black
class Gimbal3D(ThreeDScene):
    def construct(self):
        # Set camera view
        self.camera.frame.reorient(phi_degrees = 70, theta_degrees = 50)   

        # Add legend and coordinate frame (labels will remain fixed on screen)
        self._add_legend()

        # Create the rings and arrow
        outer_ring, middle_ring, inner_ring, arrow = self._create_rings()
        roll = ValueTracker(0)
        roll_axis = np.array([1.0, 0.0, 0.0])

        pitch = ValueTracker(0)
        pitch_axis = np.array([0.0, 1.0, 0.0])

        yaw = ValueTracker(0)
        yaw_axis = np.array([0.0, 0.0, 1.0])

        prev_ring_angles = np.array([0.0, 0.0, 0.0])

        texts = self._display_commands(roll.get_value(), pitch.get_value(), yaw.get_value())
        texts[0].add_updater(lambda m: m.set_value(roll.get_value()))
        texts[1].add_updater(lambda m: m.set_value(pitch.get_value()))
        texts[2].add_updater(lambda m: m.set_value(yaw.get_value()))

        # Update for all rings
        def update_rings(m):
            diff_angles = np.array([roll.get_value(), pitch.get_value(), yaw.get_value()]) - prev_ring_angles
            # diff_angles = np.vectorize(radians)(diff_angles)

            # Roll
            outer_ring.rotate(diff_angles[0], axis=roll_axis)
            middle_ring.rotate(diff_angles[[0]], axis=roll_axis)
            inner_ring.rotate(diff_angles[0], axis=roll_axis)
            arrow.rotate(diff_angles[0], axis=roll_axis)
            pitch_axis[:] = np.array(so3.apply(so3.from_axis_angle((list(roll_axis), diff_angles[0])), list(pitch_axis)))
            yaw_axis[:] = np.array(so3.apply(so3.from_axis_angle((list(roll_axis), diff_angles[0])), list(yaw_axis)))

            # Pitch
            middle_ring.rotate(diff_angles[1], axis=pitch_axis)
            inner_ring.rotate(diff_angles[1], axis=pitch_axis)
            arrow.rotate(diff_angles[1], axis=pitch_axis)
            yaw_axis[:] = np.array(so3.apply(so3.from_axis_angle((list(pitch_axis), diff_angles[1])), list(yaw_axis)))
            
            # Yaw
            inner_ring.rotate(diff_angles[2], axis=yaw_axis)
            arrow.rotate(diff_angles[2], axis=yaw_axis)

            prev_ring_angles[:] = np.array([roll.get_value(), pitch.get_value(), yaw.get_value()])
            # Hacky fix to keep the arrow in the center. Arrow seems to be drifting for some reason
            arrow.move_to(inner_ring.get_center())
            
        arrow.add_updater(update_rings)

        # Show gimbal lock
        self.play(ApplyMethod(pitch.increment_value, -math.pi/2), run_time=2, rate_func=linear)
        self.play(ApplyMethod(roll.increment_value, 2 * math.pi), ApplyMethod(yaw.increment_value, 2 * math.pi), run_time=4, rate_func=linear)
        
    def _create_rings(self):
        # Parameters
        ring_thickness = 0.13  # Tube (minor) radius

        # Outer Torus (Roll ring)
        outer_ring_sphere = Sphere(radius=0.2, color=RED)
        outer_ring_ob = Torus(r1=3, r2=ring_thickness, color=RED)
        outer_ring_sphere.move_to(outer_ring_ob.get_center() + np.array([-3, 0, 0]))
        outer_ring = Group(outer_ring_ob, outer_ring_sphere)
        outer_ring.rotate(PI / 2, axis=UP)

        # Middle Torus (Pitch ring)
        middle_ring_sphere = Sphere(radius=0.2, color=GREEN)
        middle_ring_ob = Torus(r1=2.5, r2=ring_thickness, color=GREEN)
        middle_ring_sphere.move_to(middle_ring_ob.get_center() + np.array([0, 2.5, 0]))
        middle_ring = Group(middle_ring_ob, middle_ring_sphere)
        middle_ring.rotate(PI / 2, axis=RIGHT)

        # Inner Torus (Yaw ring)
        inner_ring_sphere = Sphere(radius=0.2, color=BLUE)
        inner_ring_ob = Torus(r1=2.1, r2=ring_thickness, color=BLUE)
        inner_ring_sphere.move_to(inner_ring_ob.get_center() + np.array([2.1, 0, 0]))
        inner_ring = Group(inner_ring_ob, inner_ring_sphere)

        # Center the rings together
        middle_ring.move_to(outer_ring.get_center())
        inner_ring.move_to(outer_ring.get_center())

        arrow = self._create_center_arrow()
        arrow.move_to(outer_ring.get_center())

        self.add(outer_ring, middle_ring, inner_ring)

        return outer_ring, middle_ring, inner_ring, arrow
    
    def _display_coordinate_frame(self, origin=ORIGIN, axis_length=1.0, axis_thickness=0.05):
        """Create a 3D coordinate frame with X, Y, and Z axes."""
        x_axis = Line3D(start=origin, end=origin + np.array([axis_length, 0, 0]), width=axis_thickness, color=RED)
        y_axis = Line3D(start=origin, end=origin + np.array([0, axis_length, 0]), width=axis_thickness, color=GREEN)
        z_axis = Line3D(start=origin, end=origin + np.array([0, 0, axis_length]), width=axis_thickness, color=BLUE)
        
        # Labels
        x_label = Text("X", color=RED).scale(0.5).next_to(x_axis.get_end(), RIGHT, buff=0.1)
        y_label = Text("Y", color=GREEN).scale(0.5).next_to(y_axis.get_end(), UP, buff=0.1)
        z_label = Text("Z", color=BLUE).scale(0.5).next_to(z_axis.get_end(), OUT, buff=0.1)

        # Group all elements
        frame = Group(x_axis, y_axis, z_axis, x_label, y_label, z_label)
        self.add(frame)

        return frame
        
    def _add_legend(self):
        legend_items = VGroup(
            self._legend_entry(RED, "Roll (X-axis)"),
            self._legend_entry(GREEN, "Pitch (Y-axis)"),
            self._legend_entry(BLUE, "Yaw (Z-axis)")
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.3).to_corner(UL)

        legend_items.fix_in_frame()
        self.add(legend_items)

    def _legend_entry(self, color, label_text):
        """Helper method to create a color dot and label pair."""
        color_box = Circle(radius=0.15, fill_opacity=1, color=color, stroke_color=color).set_fill(color, opacity=1)
        label = Text(label_text, font_size=24, fill_color=BLACK).next_to(color_box, RIGHT, buff=0.2)
        return VGroup(color_box, label)
    
    def _display_commands(self, roll, pitch, yaw):
        roll_command = Text("Roll: ", font_size=30, fill_opacity=1.0, fill_color=BLACK)
        roll_number = DecimalNumber(roll, num_decimal_places=2, font_size=30, fill_opacity=1.0, fill_color=BLACK)
        roll_command = VGroup(roll_command, roll_number).arrange(RIGHT, buff=0.1)

        pitch_command = Text("Pitch: ", font_size=30, fill_opacity=1.0, fill_color=BLACK)
        pitch_number = DecimalNumber(pitch, num_decimal_places=2, font_size=30, fill_opacity=1.0, fill_color=BLACK)
        pitch_command = VGroup(pitch_command, pitch_number).arrange(RIGHT, buff=0.1)

        yaw_command = Text("Yaw: ", font_size=30, fill_opacity=1.0, fill_color=BLACK)
        yaw_number = DecimalNumber(yaw, num_decimal_places=2, font_size=30, fill_opacity=1.0, fill_color=BLACK)
        yaw_command = VGroup(yaw_command, yaw_number).arrange(RIGHT, buff=0.1)

        command_group = VGroup(roll_command, pitch_command, yaw_command).arrange(DOWN, aligned_edge=LEFT, buff=0.3).to_corner(UR)

        for command in command_group:
            command.fix_in_frame()
            self.add(command)

        return roll_number, pitch_number, yaw_number
    
    def _create_center_arrow(self):
        line = Prism(width=3.0, height=0.2, depth=0.2, color=GREY)
        cone = Cone(radius=0.3, height=0.5, color=GREY)
        cone.move_to([1.5, 0.0, 0.0])
        cone.rotate(PI / 2, axis=UP)
        arrow = Group(line, cone)
        self.add(arrow)
        return arrow