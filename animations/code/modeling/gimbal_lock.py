from manimlib import *
import numpy as np
from klampt.math import so3

manim_config.camera.background_color = "#000000"  # Set the background color to black
class Gimbal3D(ThreeDScene):
    def construct(self):
        # Set camera view
        self.camera.frame.reorient(phi_degrees = 70, theta_degrees = 50)   

        # Add legend and coordinate frame (labels will remain fixed on screen)
        self._add_legend()
        # self._display_coordinate_frame(np.array([-2, -5, -2]))

        # Define display commands: each command is a pair [strings, numeric angles]
        display_commands = [
            [["0", "-\\pi/2", "0"], [0, -PI/2, 0]],
            [["\\pi/2", "-\\pi/2", "\\pi/2"], [PI/2, -PI/2, PI/2]],
            [["\\pi/3", "-\\pi/2", "\\pi/3"], [PI/3, -PI/2, PI/3]]
        ]

        for i in range(len(display_commands)):
            # Create the rings and arrow
            outer_ring, middle_ring, inner_ring, arrow = self._create_rings()
            command = display_commands[i]
            command_text = self._display_commands(command[0])
            command_nums = command[1]
            
            ## Animations
            # Roll: rotate all components around RIGHT (X-axis)
            self.wait(0.5)
            self.play(
                command_text[0].animate.set_opacity(1),
                Rotate(outer_ring, angle=command_nums[0], axis=RIGHT, run_time=2),
                Rotate(middle_ring, angle=command_nums[0], axis=RIGHT, run_time=2), 
                Rotate(inner_ring, angle=command_nums[0], axis=RIGHT, run_time=2), 
                Rotate(arrow, angle=command_nums[0], axis=RIGHT, run_time=2),
            )

            # Pitch: rotate remaining parts around the OUT axis (camera's depth direction)
            self.wait(0.5)
            applied_rotation = so3.from_axis_angle(([1.0, 0, 0], command_nums[0]))
            new_pitch_axis = np.array(so3.apply(applied_rotation, [0, 1, 0]))
            self.play(
                command_text[0].animate.set_opacity(0.4),
                command_text[1].animate.set_opacity(1),
                Rotate(middle_ring, angle=command_nums[1], axis=new_pitch_axis, run_time=2),
                Rotate(inner_ring, angle=command_nums[1], axis=new_pitch_axis, run_time=2),
                Rotate(arrow, angle=command_nums[1], axis=new_pitch_axis, run_time=2),
            )

            # Yaw: rotate around the new Z-axis (camera's depth direction)
            self.wait(0.5)
            applied_rotation = so3.mul(so3.from_axis_angle((list(new_pitch_axis), command_nums[1])), applied_rotation)
            new_yaw_axis = np.array(so3.apply(applied_rotation, [0, 0, 1]))
            self.play(
                command_text[1].animate.set_opacity(0.4),
                command_text[2].animate.set_opacity(1),
                Rotate(inner_ring, angle=command_nums[2], axis=new_yaw_axis, run_time=2),
                Rotate(arrow, angle=command_nums[2], axis=new_yaw_axis, run_time=2)
            )     

            if i != len(display_commands) - 1:
                self.play(
                    FadeOut(outer_ring),
                    FadeOut(middle_ring),
                    FadeOut(inner_ring),
                    FadeOut(arrow),
                    *[FadeOut(item) for item in command_text],
                )
        
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
        label = Text(label_text, font_size=24, color=color).next_to(color_box, RIGHT, buff=0.2)
        return VGroup(color_box, label)
    
    def _display_commands(self, commands):
        roll_command = Tex(f"Roll: {commands[0]}", font_size=35, fill_opacity=0.5)
        pitch_command = Tex(f"Pitch: {commands[1]}", font_size=35, fill_opacity=0.5)
        yaw_command = Tex(f"Yaw: {commands[2]}", font_size=35, fill_opacity=0.5)

        command_group = VGroup(roll_command, pitch_command, yaw_command).arrange(DOWN, aligned_edge=LEFT, buff=0.3).to_corner(UR)

        for command in command_group:
            command.fix_in_frame()
            self.add(command)

        return command_group
    
    def _create_center_arrow(self):
        line = Prism(width=3.0, height=0.2, depth=0.2, color=WHITE)
        cone = Cone(radius=0.3, height=0.5, color=WHITE)
        cone.move_to([1.5, 0.0, 0.0])
        cone.rotate(PI / 2, axis=UP)
        arrow = Group(line, cone)
        self.add(arrow)
        return arrow