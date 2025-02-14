from manim import *

class NewtonsMethod1D(Scene):
    def construct(self):
        axes = Axes(
            x_range=[-0.5, 2.5, 0.5],
            y_range=[-1.5, 3.5, 1],
            x_length=6,
            y_length=4,
            tips=True
        )

        # Functions for Newton's method
        def f(x):
            return x**2 - 1

        def df(x):
            return 2*x

        # Plot the function on our axes.
        function_graph = axes.plot(f, color=BLUE)
        function_label = MathTex("f(x) = x^2 - 1").next_to(axes, UP)

        x0 = 2.0  # Starting guess
        num_iterations = 4

        x_current = x0
        prev_tangent_line = None
        prev_intersection_dot = None
        prev_intersection_label = None
        prev_dashed_line = None
        
        dot = Dot(color=YELLOW).move_to(axes.coords_to_point(x0, 0))

        # Add the axes, function graph, and initial dot to the scene.
        self.play(
            Create(axes),
            Create(function_graph),
            Write(function_label),
            FadeIn(dot, scale=0.5)
        )
        self.wait()

        for i in range(num_iterations):
            ######################################
            # Compute
            ######################################

            # Current x_n
            intersection_dot = Dot(
                axes.coords_to_point(x_current, 0),
                color=MAROON
            )
            intersection_label = MathTex(f"x_{i}").next_to(intersection_dot, DOWN)

            # Current point (x_n, f(x_n))
            fx_val = f(x_current)
            dfx_val = df(x_current)

            # Dashed Line
            dashed_line = DashedLine(
                start=axes.coords_to_point(x_current, 0), 
                end=axes.coords_to_point(x_current, fx_val),
                dash_length=0.1, dashed_ratio=0.5  # Adjusts spacing
            )
            
            # Compute the next iteration:
            # x_{n+1} = x_n - f(x_n)/f'(x_n)
            x_next = x_current - fx_val / dfx_val

            # Create a tangent line at (x_n, f(x_n)).
            # Equation of the tangent line:
            # y - f(x_n) = f'(x_n) * (x - x_n)
            # We'll find two endpoints for a line segment that extends across the axes view.
            x_min, x_max = axes.x_range[0], axes.x_range[1]
            y_min = fx_val + dfx_val * (x_min - x_current)
            y_max = fx_val + dfx_val * (x_max - x_current)
            tangent_line = Line(
                axes.coords_to_point(x_min, y_min),
                axes.coords_to_point(x_max, y_max),
                color=RED
            )

            ######################################
            # Animate
            ######################################
            self.play(FadeIn(intersection_dot), FadeIn(intersection_label))
            if i != 0:
                self.play(FadeOut(prev_tangent_line), FadeOut(prev_intersection_dot), FadeOut(prev_intersection_label), FadeOut(prev_dashed_line))
            move_up_to_new_point = dot.animate.move_to(axes.coords_to_point(x_current, fx_val))
            self.play(move_up_to_new_point, Create(dashed_line))


            tangent_line_anim = Create(tangent_line)
            self.play(tangent_line_anim)
            self.wait(0.5)
            move_down_to_axis = dot.animate.move_to(axes.coords_to_point(x_next, 0))
            self.play(move_down_to_axis)

            prev_tangent_line = tangent_line
            prev_intersection_dot = intersection_dot
            prev_intersection_label = intersection_label
            prev_dashed_line = dashed_line
            
            # Update x_current
            x_current = x_next

        # Finally, hold the scene
        self.wait(2)