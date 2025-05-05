from manimlib import *

manim_config.camera.background_color = "#FFFFFF"
class NewtonsMethod1D(Scene):
    def construct(self):
        # Set graph plane
        plane = NumberPlane(
            x_range=(-1, 3, 1),
            y_range=(-2, 4, 1),
            height=5,
            width=5,
            background_line_style={
                "stroke_width": 0  # Hides the grid lines
            },
            axis_config={
                "stroke_color": BLACK,
                "include_tip": True,
                "include_ticks": True
            }
        )
        plane.add_coordinate_labels()
        plane.shift(LEFT * 3)

        # Functions
        def f(x):
            return x**2 - 1
        
        def f_zero():
            return 1

        def df(x):
            return 2 * x

        # Function graph
        graph = plane.get_graph(f, color=BLUE)
        label = Tex(r"f(x) = x^2 - 1", fill_color=BLACK).next_to(plane, UP, buff=0.5)
        x_label = Tex("x", fill_color=BLACK).scale(0.7).next_to(plane.axes[0].get_end(), RIGHT, buff=0.2)
        y_label = Tex("y", fill_color=BLACK).scale(0.7).next_to(plane.axes[1].get_end(), UP, buff=0.2)
        plane_label = VGroup(label, x_label, y_label)

        # Error plane
        error_plane = NumberPlane(
            x_range=(-1, 5, 1),
            y_range=(-10, 1, 2),
            height=5,
            width=6,
            background_line_style={
                "stroke_width": 0  # Hides the grid lines
            },
            axis_config={
                "stroke_color": BLACK,
                "include_tip": True,
                "include_ticks": True
            },
        )

        error_plane.shift(RIGHT * 3.3)
        label = TexText(r"\textit{Error Graph}", fill_color=BLACK).next_to(error_plane, UP, buff=0.5)
        x_label = Tex("Iterations", fill_color=BLACK).scale(0.7).next_to(error_plane.axes[0].get_center(), UP, buff=0.5)
        y_label = Tex("Error", fill_color=BLACK).scale(0.7).rotate(PI/2).next_to(error_plane.axes[1].get_center(), LEFT, buff=1.0)
        error_label = VGroup(label, x_label, y_label)
        y_axis_labels = {
             -2: "10^{-2}",
             -4: "10^{-4}",
             -6: "10^{-6}",
             -8: "10^{-8}",
        }

        for y_val, label_text in y_axis_labels.items():
            tick = error_plane.y_axis.number_to_point(y_val)
            label = Tex(rf"{label_text}", fill_color=BLACK, font_size=30).next_to(tick, LEFT, buff=0.2)
            error_label.add(label)

        x0 = 2.0
        x_current = x0
        num_iterations = 5

        dot = Dot(fill_color=YELLOW, z_index=1).move_to(plane.c2p(x_current, 0))
        
        self.play(FadeIn(plane), 
                  ShowCreation(graph), 
                  Write(plane_label), 
                  FadeIn(dot, scale=0.5), 
                  FadeIn(error_plane), 
                  Write(error_label), 
                  run_time=2.0)
        self.wait()

        prev_tangent_line = None
        prev_intersection_dot = None
        prev_intersection_label = None
        prev_dashed_line = None

        for i in range(num_iterations):
            fx = f(x_current)
            dfx = df(x_current)
            x_next = x_current - fx / dfx

            # Dot and label on x-axis
            intersection_dot = Dot(plane.c2p(x_current, 0), fill_color=MAROON, z_index=2)
            intersection_label = Tex(rf"x_{i}", font_size=30, fill_color=BLACK).next_to(intersection_dot, DOWN)

            # Dashed vertical line
            dashed_line = DashedLine(
                start=plane.c2p(x_current, 0),
                end=plane.c2p(x_current, fx),
                dash_length=0.1,
                stroke_color=BLACK
            )

            # Tangent line: y - fx = dfx(x - x_current)
            x_min, x_max = -0.5, 2.5
            y_min = fx + dfx * (x_min - x_current)
            y_max = fx + dfx * (x_max - x_current)
            tangent_line = Line(
                plane.c2p(x_min, y_min),
                plane.c2p(x_max, y_max),
                color=RED
            )

            # Animate step
            error = Dot(fill_color=RED, point=error_plane.c2p(i, math.log10(x_current - f_zero())))
            self.play(FadeIn(intersection_dot), FadeIn(intersection_label), FadeIn(error))
            if i > 0:
                self.play(
                    FadeOut(prev_tangent_line),
                    FadeOut(prev_intersection_dot),
                    FadeOut(prev_intersection_label),
                    FadeOut(prev_dashed_line)
                )
            self.play(dot.animate.move_to(plane.c2p(x_current, fx)), ShowCreation(dashed_line))
            self.play(ShowCreation(tangent_line))
            self.wait(0.5)
            self.play(dot.animate.move_to(plane.c2p(x_next, 0)))

            # Save references for next iteration cleanup
            prev_tangent_line = tangent_line
            prev_intersection_dot = intersection_dot
            prev_intersection_label = intersection_label
            prev_dashed_line = dashed_line

            x_current = x_next

        self.wait(2)