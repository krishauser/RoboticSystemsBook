import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML,display

def display_animation(update,num_frames,framerate,fig=None,format='video'):
    """Plots an animation in Jupyter notebook, either as a video or an interactive animation"""
    if fig is None:
        fig = plt.gcf()
    # Create the animation
    anim = FuncAnimation(fig, update, frames=range(num_frames), interval=int(1000/framerate), repeat=False)

    if format == 'video':
        # Display the animation -- need FFMPEG
        from IPython.display import HTML,display
        try:
            display(HTML(anim.to_html5_video()))
            plt.close()
        except RuntimeError:
            #try changing the ffmpeg path for Windows
            try:
                plt.rcParams['animation.ffmpeg_path'] = 'C:/ffmpeg/bin/ffmpeg.exe'
                display(HTML(anim.to_html5_video()))
                plt.close()
            except RuntimeError:
                #fallback just plot it
                print("ffmpeg doesn't seem to be available on your system path, falling back to inline Matplotlib")
                plt.show()
    else:
        plt.show()
