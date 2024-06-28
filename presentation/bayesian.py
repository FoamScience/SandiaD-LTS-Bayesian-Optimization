from manim import *
from manim_slides import Slide
from manim.utils import color
from manim.utils.color import interpolate_color
from numpy.random import RandomState
from bayes_opt import BayesianOptimization
from bayes_opt import UtilityFunction
import numpy as np
import pandas as pd

rng = RandomState(0)
MAIN_COLOR = color.TEAL_A
BACKGROUND_COLOR = color.GRAY_E
TEXT_COLOR = color.WHITE
GRAPH_COLOR = color.BLUE_B
DOT_COLOR = color.RED_C
ITEM_ICON = "•"
very_small_size = 12.0
small_size = 16
mid_size = 20
big_size = 25
N = 6
CASE_NAME="SandiaFlameD"
Text.set_default(font="Comic Code Ligatures", color=TEXT_COLOR, font_size=small_size)
Tex.set_default(color=TEXT_COLOR, font_size=small_size)
Dot.set_default(radius=0.07, color=DOT_COLOR)

def z(x,k,m,lb):
    cond= np.abs(x)/k - np.floor(np.abs(x)/k)
    return [1-m+(m/lb)*i if i<lb else 1-m+(m/(1-lb))*(1-i) for i in cond]

def F1(x,k,m,lb):
    c=z(x,k,m,lb)
    p=(x-40)*(x-185)*x*(x+50)*(x+180)
    return 3e-9*np.abs(p)*c+10*np.abs(np.sin(0.1*x))

def keep_only_objects(slide, grp):
    slide.clear()
    for _ in grp:
        slide.add(_)

class Bayes(Slide):

    def itemize(self, items, anchor, distance, stepwise, **kwargs):
        anims = []
        mobjs = []
        for i in range(len(items)):
            mobjs.append(Text(f"{i+1}{ITEM_ICON} {items[i]}", font_size=small_size, **kwargs))
            if i == 0:
                mobjs[i].next_to(anchor, DOWN*distance).align_to(anchor, LEFT)
            else:
                mobjs[i].next_to(mobjs[i-1], DOWN).align_to(mobjs[i-1], LEFT)
        anims = [Create(mobjs[i]) for i in range(len(items))]
        if stepwise:
            for a in anims:
                self.play(a)
        else:
            self.play(AnimationGroup(*anims))
        return mobjs[-1]

    def hi_yaml(self, items, indents, anchor, distance):
        anims = []
        mobjs = []
        for i in range(len(items)):
            mobjs.append(Text(f"{items[i][0]}: {items[i][1]}", font_size=small_size,
                 t2w={f"{items[i][0]}:": BOLD}, t2c={f"{items[i][0]}:": GREEN}))
            if i == 0:
                mobjs[i].next_to(anchor, DOWN*distance).align_to(anchor, LEFT).shift(indents[i]*RIGHT)
            else:
                mobjs[i].next_to(mobjs[i-1], DOWN).align_to(mobjs[i-1], LEFT).shift((indents[i]-indents[i-1])*RIGHT)
        anims = [Create(mobjs[i]) for i in range(len(items))]
        self.play(AnimationGroup(*anims))
        return mobjs[-1]

    def construct(self):
        self.camera.background_color = BACKGROUND_COLOR
        # Title page
        layout = Group()
        title = Text(f"Optimization of combustion processes with Bayesian algorithms", font_size=big_size)#.to_edge(UP+LEFT)
        footer = Text("NHR4CES - Machine learning for combustion workshop", t2w={"NHR4CES": BOLD}, font_size=very_small_size).to_edge(DOWN+RIGHT)
        author = Text("Mohammed Elwardi Fadeli, Nov. 2023", font_size=very_small_size).to_edge(DOWN+LEFT)
        logo = ImageMobject("./images/nhr-tu-logo.png").next_to(title, UP).scale(0.6)#.to_edge(UP+RIGHT)
        layout.add(title, footer, author, logo)
        self.play(FadeIn(layout))
        self.next_slide()

        t1 = Text(f"Motivation - GAs", font_size=big_size).to_edge(UP+LEFT)

        anims =[
            title.animate.to_edge(UP+LEFT),
            Transform(title, t1),
            Transform(logo, logo.copy().scale(0.5).to_edge(UP+RIGHT)),
        ]
        self.play(AnimationGroup(*anims))

        ### SECTION MOTIVATION

        # Function object plot
        graphs = VGroup()
        grid = Axes(x_range=[-200, 200, 20], y_range=[0, 200, 20],
                    x_length=10, y_length=5, tips=False,
                    axis_config={"color": interpolate_color(BACKGROUND_COLOR, WHITE, 0.5)},)
        graphs += grid
        graphs += grid.plot(lambda x: F1(np.array([x]), 1, 0, 0.01)[0], color=GRAPH_COLOR)
        graphs += Text(r"Objective Function to minimize", font_size=very_small_size).next_to(grid, UP)
        graphs += Text(r"Opt. param.", font_size=very_small_size).next_to(grid, DOWN+RIGHT).shift(LEFT)
        graphs.to_edge(UP)

        # Dots for population members
        pop = [Dot().move_to(LEFT*rng.uniform(4.3,5.2)+DOWN*rng.uniform(2.3,2.6)) for _ in range(N)]
        for d in pop:
            graphs.add(d)
        pop_text = Text("Population: 0").move_to(LEFT*4.5+DOWN*2.8)
        graphs.add(pop_text)
        self.play(DrawBorderThenFill(graphs))
        self.next_slide()

        ev = Text(r"0. Generate random initial population of chromosomes", font_size=small_size).next_to(grid, DOWN)
        self.play(FadeIn(ev))
        self.wait(2)
        self.play(FadeOut(ev))
        self.remove(ev)

        # Random selection of initial population
        xs = [rng.uniform(-200, 200) for _ in pop]
        pos = [grid.c2p(x,0) for x in xs]
        anims = [dot.animate.move_to(target) for dot, target in zip(pop, pos)]
        self.play(AnimationGroup(*anims))
        self.next_slide()

        ev = Text(r"1. Evaluate objective functions", font_size=small_size).next_to(grid, DOWN)
        self.play(FadeIn(ev))
        self.wait(2)
        self.play(FadeOut(ev))
        self.remove(ev)

        # Move population to the plot, and color-grade them
        pos = [grid.c2p(x, F1(np.array([x]), 1,0,0.01)[0]) for x in xs]
        anims = [dot.animate.move_to(target) for dot, target in zip(pop, pos)]
        self.play(AnimationGroup(*anims))
        colors = [interpolate_color(DOT_COLOR, WHITE, F1(np.array([x]), 1,0,0.01)[0]/180.0) for x in xs]
        anims = [dot.animate.set_color(c) for dot, c in zip(pop, colors)]
        self.play(AnimationGroup(*anims))
        self.next_slide()

        ev = Text(r"2. Select the better-performing chromosome", font_size=small_size).next_to(grid, DOWN)
        self.play(FadeIn(ev))
        self.wait(2)
        self.play(FadeOut(ev))
        self.remove(ev)

        # Draw a "fictional" selection line
        lpos = [grid.c2p(-200, 50), grid.c2p(200, 50)]
        line = DashedLine(lpos[0], lpos[1], color=GREEN)
        self.play(Create(line))

        # Keep only best performing members
        worst_x = [i for i in range(len(xs)) if F1(np.array([xs[i]]), 1,0,0.01)[0] > 50]
        marks = [Circle(radius=0.5, color=colors[i]).move_to(pos[i]) for i in worst_x]
        anims_1 = [Broadcast(m, focal_point=m.get_center()) for m in marks]
        anims_2 = [FadeOut(pop[e]) for e in worst_x]
        anims = [*anims_1, *anims_2]
        self.play(AnimationGroup(*anims))
        self.play(FadeOut(line))
        self.next_slide()

        ev = Text(r"3. Perform cross-overs between selected chromosomes", font_size=small_size).next_to(grid, DOWN)
        self.play(FadeIn(ev))
        self.wait(2)
        self.play(FadeOut(ev))
        self.remove(ev)

        # Do cross-over animation
        best_x = [i for i in range(len(xs)) if i not in worst_x]
        e1 = best_x[rng.randint(0, len(best_x))]
        e2 = best_x[rng.randint(0, len(best_x))]
        while e1 == e2:
            # May hang if only one element remains!!
            e1 = best_x[rng.randint(0, len(best_x))]
        marks = [Circle(radius=0.5, color=colors[e]).move_to(pos[e]) for e in [e1, e2]]
        anims = [Broadcast(m, focal_point=m.get_center()) for m in marks]
        self.play(AnimationGroup(*anims))
        middle = (pos[e1]+pos[e2])/2
        middle[1] = 0
        originals = [e.get_center() for e in pop]
        anims = [dot.animate.move_to(middle) for dot in [pop[e1], pop[e2]]]
        self.play(AnimationGroup(*anims))
        new_x = [rng.uniform(-200, grid.p2c(middle)[0]), rng.uniform(-200, 200), 2, rng.uniform(grid.p2c(middle)[0], 200)]
        anims_1 = [pop[e].animate.move_to(originals[e]) for e in [e1, e2]]
        old_len = len(pop)
        new_pos = [grid.c2p(x,0) for x in new_x]
        for e in new_pos:
            pop.append(Dot(color=interpolate_color(DOT_COLOR, YELLOW, 0.8)).move_to(middle))
        for d in pop[old_len:]:
            graphs.add(d)
        new_pos = [grid.c2p(x,0) for x in new_x]
        anims_2 = [dot.animate.move_to(target) for dot, target in zip(pop[old_len:], new_pos)]
        anims = [*anims_1, *anims_2]
        self.wait(1)
        self.play(AnimationGroup(*anims))
        self.next_slide()

        ev = Text(r"4.1. Most offsprings get evaluated 'as-is'", font_size=small_size).next_to(grid, DOWN)
        self.play(FadeIn(ev))
        self.wait(2)
        self.play(FadeOut(ev))
        self.remove(ev)

        new_pos = [grid.c2p(new_x[i], F1(np.array([new_x[i]]), 1,0,0.01)[0]) if i != 0 else grid.c2p(-20, 0) for i in range(len(new_x))]
        anims = [dot.animate.move_to(target) for dot, target in zip(pop[old_len+1:], new_pos[1:])]
        self.play(AnimationGroup(*anims))
        self.next_slide()

        ev = Text(r"4.2. But some will be mutated", font_size=small_size).next_to(grid, DOWN)
        self.play(FadeIn(ev))
        self.wait(2)
        self.play(FadeOut(ev))
        self.remove(ev)

        self.play(pop[old_len].animate.move_to(new_pos[0]))
        new_pos[0] = grid.c2p(-20, F1(np.array([-20]),1,0,0.01)[0])
        self.play(pop[old_len].animate.move_to(new_pos[0]))

        new_pop_text = Text("Population: 1").move_to(LEFT*4.5+DOWN*2.8)
        self.play(Transform(pop_text, new_pop_text))

        coords = [e.get_center()[1] for e in pop]
        min_value = min(coords)
        max_value = max(coords)
        normalized = [(x - min_value) / (max_value - min_value) for x in coords]
        colors = [interpolate_color(DOT_COLOR, WHITE, y) for y in normalized]
        pop_sub = [pop[i] for i in best_x]
        colors_sub = [colors[i] for i in best_x]
        anims = [dot.animate.set_color(c) for dot, c in zip([*pop_sub, *pop[old_len:]], [*colors_sub, *colors[old_len:]])]
        self.play(AnimationGroup(*anims))

        ev = Text(r"Next generation of the population is ready!", font_size=small_size).next_to(grid, DOWN)
        self.play(FadeIn(ev))
        self.wait(2)
        self.play(FadeOut(ev))
        self.remove(ev)

        ev = Text(r"Repeat until global minima is found!", t2w={'global':BOLD}, font_size=small_size).next_to(grid, DOWN)
        self.play(FadeIn(ev))
        self.wait(2)
        self.play(FadeOut(ev))
        self.remove(ev)

        self.wait()
        self.next_slide()  # Waits user to press continue to go to the next slide
        keep_only_objects(self, layout)

        pros = Text("- What's so good about it?", font_size=mid_size).next_to(title, DOWN*2).align_to(title, LEFT)
        self.play(Create(pros))

        items = [
            "Naturally parallelizable! even with different evolutionary algorithms.",
            "More than enough freedom in exploring the search space.",
            "Can handle noisy objective functions well enough.",
            "Has no trouble with hard constraints."
        ]
        last = self.itemize(items, pros, 1.5, True,
            t2w={f"1{ITEM_ICON}": BOLD, f"2{ITEM_ICON}": BOLD, f"3{ITEM_ICON}": BOLD, f"4{ITEM_ICON}": BOLD},
            t2c={f"1{ITEM_ICON}": GREEN, f"2{ITEM_ICON}": GREEN, f"3{ITEM_ICON}": GREEN, f"4{ITEM_ICON}": GREEN})

        self.next_slide()
        cons = Text("- What's not so good?", font_size=mid_size).next_to(last, DOWN*2).align_to(last, LEFT)
        self.play(Create(cons))

        items = [
            "Slow to converge in terms of number of expensive objective function evaluations.",
            "Too much time wasted on exploring the search space.",
            "Calls for hyperparameter tuning/optimization.",
            "'Just works' with little to no theoretical guarantees."
        ]
        last = self.itemize(items, cons, 1.5, True,
            t2w={f"1{ITEM_ICON}": BOLD, f"2{ITEM_ICON}": BOLD, f"3{ITEM_ICON}": BOLD, f"4{ITEM_ICON}": BOLD},
            t2c={f"1{ITEM_ICON}": RED, f"2{ITEM_ICON}": RED, f"3{ITEM_ICON}": RED, f"4{ITEM_ICON}": RED})

        self.wait()
        self.next_slide()  # Waits user to press continue to go to the next slide
        keep_only_objects(self, layout)

        ## SECTION BAYES OPT

        t2 = Text(f"Bayesian Optimization", font_size=big_size).to_edge(UP+LEFT)
        self.play(Transform(title, t2))

        # Function object again
        graphs = VGroup()
        grid = Axes(x_range=[-200, 200, 20], y_range=[0, 200, 20],
                    x_length=10, y_length=5, tips=False,
                    axis_config={"color": interpolate_color(BACKGROUND_COLOR, WHITE, 0.5)},)
        graphs += grid
        graphs += grid.plot(lambda x: F1(np.array([x]), 1, 0, 0.01)[0], color=GRAPH_COLOR)
        graphs += Text(r"Objective Function to minimize", font_size=very_small_size).next_to(grid, UP)
        graphs += Text(r"Opt. param.", font_size=very_small_size).next_to(grid, DOWN+RIGHT).shift(LEFT)
        graphs.to_edge(UP)
        self.play(DrawBorderThenFill(graphs))
        self.next_slide()  # Waits user to press continue to go to the next slide

        ev = Text(r"0. Generate initial samples randomly", font_size=small_size).next_to(grid, DOWN)
        self.play(FadeIn(ev))
        self.wait(1)
        self.play(FadeOut(ev))
        self.remove(ev)

        def obj_func(x):
            return -F1(np.array([x]), 1,0,0.01)[0]

        def posterior(optimizer, x, y, X):
            optimizer._gp.fit(x, y)
            mu, sigma = optimizer._gp.predict(X, return_std=True)
            return (mu, sigma)

        optimizer = BayesianOptimization(obj_func, {'x': (-200, 200)}, random_state=100)
        acq_function = UtilityFunction(kind="ei", kappa=5)
        optimizer.maximize(init_points=3, n_iter=0, acquisition_function=acq_function)

        xx = np.array([[res["params"]["x"]] for res in optimizer.res])
        yy = np.array([res["target"] for res in optimizer.res])

        sample = [e[0] for e in xx]# [xx[0][0], xx[1][0]]
        sample_dots = [Dot(color=DOT_COLOR).move_to(grid.c2p(e, 0)) for e in sample]
        self.play(*[Create(d) for d in sample_dots])
        self.next_slide()  # Waits user to press continue to go to the next slide

        ev = Text(r"1. Evaluate objective functions", font_size=small_size).next_to(grid, DOWN)
        self.play(FadeIn(ev))
        self.wait(2)
        self.play(FadeOut(ev))
        self.remove(ev)
        sample_f = [-obj_func(e) for e in sample]
        anims = [sample_dots[i].animate.move_to(grid.c2p(sample[i], sample_f[i])) for i in range(len(sample))]
        self.play(AnimationGroup(*anims))
        self.next_slide()  # Waits user to press continue to go to the next slide

        ev = Text(r"2.1 Fit a Gaussian process model", font_size=small_size).next_to(grid, DOWN)
        self.play(FadeIn(ev))
        self.wait(2)
        self.play(FadeOut(ev))
        self.remove(ev)

        gp = grid.plot(lambda x: -posterior(optimizer, xx, yy, np.array([x]).reshape(1, -1))[0][0], color=GREEN)
        graphs.add(gp)
        self.play(DrawBorderThenFill(gp))
        ev = Text("Fitted Gaussian Process model", font_size=very_small_size, color=GREEN).next_to(gp, UP)
        self.play(FadeIn(ev))
        self.wait(2)
        self.play(FadeOut(ev))
        self.remove(ev)

        self.next_slide()  # Waits user to press continue to go to the next slide
        ev = Text(r"2.1 With 95% confidence interval", font_size=small_size).next_to(grid, DOWN)
        self.play(FadeIn(ev))
        self.wait(2)
        self.play(FadeOut(ev))
        self.remove(ev)

        def confidence(optimizer, x, y, X, interval):
            mu, s = posterior(optimizer, x, y, np.array([X]).reshape(1, -1))
            return -(mu[0] + interval*s[0])

        ce11 = grid.plot(lambda x: confidence(optimizer, xx, yy, x, 1.96), color=ORANGE)
        ce12 = grid.plot(lambda x: confidence(optimizer, xx, yy, x, -1.96), color=ORANGE)
        ce1 = grid.get_area(ce11, bounded_graph=ce12, opacity=0.5, x_range=(-200, 200))
        self.play(DrawBorderThenFill(ce1))

        self.next_slide()  # Waits user to press continue to go to the next slide
        ev = Text(r"2.2 Estimate acquisition function (eg. EI)", font_size=small_size).next_to(grid, DOWN)
        self.play(FadeIn(ev))
        self.wait(2)
        self.play(FadeOut(ev))
        self.remove(ev)

        ei = grid.plot(lambda x: 100-10*acq_function.utility([[x]], optimizer._gp, 0)[0], color=ORANGE)
        self.play(FadeOut(ce1))
        ei_area = grid.get_area(ei, x_range=(-200, 200), opacity=0.3, color=[ORANGE, ORANGE])
        self.play(FadeIn(ei_area))
        ev = Text("Acquisition function guides exploration-exploitation trade-off", font_size=very_small_size, color=ORANGE).next_to(ei_area, UP)
        self.play(FadeIn(ev))
        self.wait(2)
        self.play(FadeOut(ev))
        self.remove(ev)

        self.next_slide()  # Waits user to press continue to go to the next slide
        ev = Text(r"2.3 Pick next candidate, maximzing EI", font_size=small_size).next_to(grid, DOWN)
        self.play(FadeIn(ev))
        self.wait(2)
        self.play(FadeOut(ev))
        self.remove(ev)

        niters = 20
        for i in range(niters):
            optimizer.maximize(init_points=0, n_iter=1)
            xx = np.array([[res["params"]["x"]] for res in optimizer.res])
            yy = np.array([res["target"] for res in optimizer.res])
            sample = [e[0] for e in xx]
            sample_dots.append(Dot(color=DOT_COLOR).move_to(grid.c2p(sample[-1], 0)))
            sample_f = [-obj_func(e) for e in sample]
            if i<4:
                # visualize first three
                self.play(FadeOut(ei_area), FadeOut(gp))
                graphs.add(sample_dots[-1])
                anims = [d.animate.set_color(GRAPH_COLOR) for d in sample_dots[:-1]]
                self.play(AnimationGroup(*anims))
                self.play(sample_dots[-1].animate.move_to(grid.c2p(sample[-1], sample_f[-1])))
                gp = grid.plot(lambda x: -posterior(optimizer, xx, yy, np.array([x]).reshape(1, -1))[0][0], color=GREEN)
                graphs.add(gp)
                self.play(DrawBorderThenFill(gp))
                ce11 = grid.plot(lambda x: confidence(optimizer, xx, yy, x, 1.96), color=ORANGE)
                ce12 = grid.plot(lambda x: confidence(optimizer, xx, yy, x, -1.96), color=ORANGE)
                ce1 = grid.get_area(ce11, bounded_graph=ce12, opacity=0.5, x_range=(-200, 200))
                self.play(DrawBorderThenFill(ce1))
                ei = grid.plot(lambda x: 100-10*acq_function.utility([[x]], optimizer._gp, 0)[0], color=ORANGE)
                self.play(FadeOut(ce1))
                ei_area = grid.get_area(ei, x_range=(-200, 200), opacity=0.3, color=[ORANGE, ORANGE])
                self.play(FadeIn(ei_area))
            else:
                if i == niters-1:
                    self.play(FadeOut(ei_area), FadeOut(gp))
                    self.next_slide()  # Waits user to press continue to go to the next slide
                    ev = Text(r"After some iterations...", font_size=small_size).next_to(grid, DOWN)
                    self.play(FadeIn(ev))
                    self.wait(2)
                    self.play(FadeOut(ev))
                    self.remove(ev)
                    graphs.add(*sample_dots[4:])
                    anims = [d.animate.set_color(GRAPH_COLOR) for d in sample_dots[:3]]
                    self.play(AnimationGroup(*anims))
                    anims = [sample_dots[j].animate.move_to(grid.c2p(sample[j], sample_f[j])) for j in range(4, len(sample_dots))]
                    self.play(AnimationGroup(*anims))
                    gp = grid.plot(lambda x: -posterior(optimizer, xx, yy, np.array([x]).reshape(1, -1))[0][0], color=GREEN)
                    graphs.add(gp)
                    self.play(DrawBorderThenFill(gp))
                    self.next_slide()
                    ce11 = grid.plot(lambda x: confidence(optimizer, xx, yy, x, 1.96), color=ORANGE)
                    ce12 = grid.plot(lambda x: confidence(optimizer, xx, yy, x, -1.96), color=ORANGE)
                    ce1 = grid.get_area(ce11, bounded_graph=ce12, opacity=0.5, x_range=(-200, 200))
                    self.play(DrawBorderThenFill(ce1))
                    self.next_slide()
                    ei = grid.plot(lambda x: 100-10*acq_function.utility([[x]], optimizer._gp, 0)[0], color=ORANGE)
                    self.play(FadeOut(ce1))
                    ei_area = grid.get_area(ei, x_range=(-200, 200), opacity=0.3, color=[ORANGE, ORANGE])
                    self.play(FadeIn(ei_area))
            self.next_slide()  # Waits user to press continue to go to the next slide

        keep_only_objects(self, layout)

        pros = Text("- What's so good about it?", font_size=mid_size).next_to(title, DOWN*2).align_to(title, LEFT)
        self.play(Create(pros))

        items = [
            "Faster convergence, less expensive objective function evaluations.",
            "Effective for hyperparameter optimization",
            "Can handle noisy/stochastic objective functions well enough.",
            "Ending up with trained surrogate models, including uncertainty info!"
        ]
        last = self.itemize(items, pros, 1.5, True,
            t2w={f"1{ITEM_ICON}": BOLD, f"2{ITEM_ICON}": BOLD, f"3{ITEM_ICON}": BOLD, f"4{ITEM_ICON}": BOLD},
            t2c={f"1{ITEM_ICON}": GREEN, f"2{ITEM_ICON}": GREEN, f"3{ITEM_ICON}": GREEN, f"4{ITEM_ICON}": GREEN})

        self.next_slide()
        cons = Text("- What's not so good?", font_size=mid_size).next_to(last, DOWN*2).align_to(last, LEFT)
        self.play(Create(cons))

        items = [
            "Easy to get stuck at local optima!",
            "Poor scalibility with high-dimensional optimization problems",
        ]
        last = self.itemize(items, cons, 1.5, True,
            t2w={f"1{ITEM_ICON}": BOLD, f"2{ITEM_ICON}": BOLD, f"3{ITEM_ICON}": BOLD, f"4{ITEM_ICON}": BOLD},
            t2c={f"1{ITEM_ICON}": RED, f"2{ITEM_ICON}": RED, f"3{ITEM_ICON}": RED, f"4{ITEM_ICON}": RED})

        self.wait()
        self.next_slide()  # Waits user to press continue to go to the next slide
        keep_only_objects(self, layout)

        pros = Text("- Acquisition functions?", font_size=mid_size).next_to(title, DOWN*2).align_to(title, LEFT)
        self.play(Create(pros))

        items = [
            "Probability of Improvement (PI) -> max. probability of exceeding a performance threshold",
            "Expected Improvement (EI) -> max. expected improvement over current best observation",
            "Upper Confidence Bound (UCB) -> balance exploration and exploitation",
            "Knowledge Gradient (KG) -> very expensive FOs, gain max information"
        ]
        last = self.itemize(items, pros, 1.5, True,
            t2w={f"1{ITEM_ICON}": BOLD, f"2{ITEM_ICON}": BOLD, f"3{ITEM_ICON}": BOLD, f"4{ITEM_ICON}": BOLD},
            t2c={f"1{ITEM_ICON}": GREEN, f"2{ITEM_ICON}": GREEN, f"3{ITEM_ICON}": GREEN, f"4{ITEM_ICON}": GREEN})

        self.next_slide()
        cons = Text("- The Gaussian Process model?", font_size=mid_size).next_to(last, DOWN*2).align_to(last, LEFT)
        self.play(Create(cons))

        items = [
            "Standard GP -> Assume the underlying function is a sample from a Gaussian process",
            "Sparse GP -> Generally for reducing the computational complexity",
            "Multi-Output GP -> For multi-objective opt., capturing dependencies between objectives",
        ]
        last = self.itemize(items, cons, 1.5, True,
            t2w={f"1{ITEM_ICON}": BOLD, f"2{ITEM_ICON}": BOLD, f"3{ITEM_ICON}": BOLD, f"4{ITEM_ICON}": BOLD},
            t2c={f"1{ITEM_ICON}": RED, f"2{ITEM_ICON}": RED, f"3{ITEM_ICON}": RED, f"4{ITEM_ICON}": RED})

        self.next_slide()
        keep_only_objects(self, layout)

        ## SECTION MULTI-OBJ BAYES OPT

        t3 = Text(f"Multi-Obj. Bayesian Optimization", font_size=big_size).to_edge(UP+LEFT)
        self.play(Transform(title, t3))

        pros = Text("- Pareto-frontier for decision-making", font_size=mid_size).next_to(title, DOWN*2).align_to(title, LEFT)
        self.play(Create(pros))

        items = [
            "Optimize conflicting objectives simultaneously",
            "Acquisition functions are important!",
        ]
        last = self.itemize(items, pros, 1.5, True,
            t2w={f"1{ITEM_ICON}": BOLD, f"2{ITEM_ICON}": BOLD, f"3{ITEM_ICON}": BOLD, f"4{ITEM_ICON}": BOLD},
            t2c={f"1{ITEM_ICON}": GREEN, f"2{ITEM_ICON}": GREEN, f"3{ITEM_ICON}": GREEN, f"4{ITEM_ICON}": GREEN})

        self.next_slide()
        grid = Axes(x_range=[2, 10, 2], y_range=[2, 10, 2],
                    x_length=5, y_length=5, tips=False,
                    axis_config={"color": interpolate_color(BACKGROUND_COLOR, WHITE, 0.5)},).to_edge(RIGHT)
        graphs = VGroup()
        graphs += grid
        #graphs += grid.plot(lambda x: 20/x, color=GRAPH_COLOR)
        graphs += Text(r"Objective 1", font_size=very_small_size).next_to(grid, 0.5*DOWN)
        graphs += Text(r"Objective 2", font_size=very_small_size).next_to(grid, LEFT).rotate(PI/2).shift(0.5*RIGHT)
        dots = [Dot(color=DOT_COLOR).move_to(grid.c2p(x, 20.0/x)) for x in np.linspace(2.5, 9.5, 6)]
        graphs.add(*dots)
        self.play(DrawBorderThenFill(graphs))

        opt = grid.plot(lambda x: 20.0/x, color=RED)
        opt_area = grid.get_area(opt, x_range=(2, 10), opacity=0.3, color=[RED, RED])
        self.play(FadeIn(opt_area))

        arrows = [Arrow(dots[3].get_center()+3*UP, d.get_center(), color=ORANGE, max_tip_length_to_length_ratio=0.04, max_stroke_width_to_length_ratio=0.3) for d in [dots[2], dots[4]]]
        self.play(AnimationGroup(*[GrowArrow(a) for a in arrows]))
        tt = Text("generated by surrogate model", font_size=very_small_size, color=ORANGE).next_to(arrows[0], UP)
        tt1 = Text("Best parameter set", font_size=very_small_size, color=ORANGE).next_to(tt, 0.5*UP)
        self.play(FadeIn(tt1, tt))

        self.next_slide()
        self.play(FadeOut(opt_area, tt1, tt), AnimationGroup(*[FadeOut(a) for a in arrows]))

        line = grid.plot(lambda x: 10, color=GREEN)
        opt_area = grid.get_area(line, x_range=(2, 10), opacity=0.3, color=[GREEN, GREEN], bounded_graph=opt)
        self.play(FadeIn(opt_area))
        dots = [Dot(color=GREEN).move_to(grid.c2p(x, y)) for x, y in zip(rng.uniform(2, 10, 10), rng.uniform(2, 10, 10)) if y >= 20/x]
        dots.append(Dot(color=GREEN).move_to(grid.c2p(5, 4)))
        self.play(AnimationGroup(*[DrawBorderThenFill(d) for d in dots]))
        
        arrows = [Arrow(dots[-1].get_center()+4.5*LEFT, d.get_center(), color=GREEN, max_tip_length_to_length_ratio=0.04, max_stroke_width_to_length_ratio=0.3) for d in [dots[2], dots[-1]]]
        self.play(AnimationGroup(*[GrowArrow(a) for a in arrows]))
        tt = Text("Trials", font_size=very_small_size, color=GREEN).next_to(arrows[1], LEFT)
        self.play(FadeIn(tt))
        
        self.next_slide()
        self.play(FadeOut(opt_area, tt))
        self.play(FadeOut(graphs), AnimationGroup(*[FadeOut(d) for d in dots]), AnimationGroup(*[FadeOut(a) for a in arrows]))

        cons = Text("- Relative feature importances", font_size=mid_size).next_to(last, DOWN*2).align_to(last, LEFT)
        self.play(Create(cons))

        items = [
            "Which parameters matter the most?",
            "What parameters affect which objectives?",
        ]
        last = self.itemize(items, cons, 1.5, True,
            t2w={f"1{ITEM_ICON}": BOLD, f"2{ITEM_ICON}": BOLD, f"3{ITEM_ICON}": BOLD, f"4{ITEM_ICON}": BOLD, "the Most": BOLD},
            t2c={f"1{ITEM_ICON}": RED, f"2{ITEM_ICON}": RED, f"3{ITEM_ICON}": RED, f"4{ITEM_ICON}": RED})

        self.next_slide()
        graphs = VGroup()
        grid = Axes(x_range=[2, 10, 2], y_range=[2, 10, 2],
                    x_length=5, y_length=5, tips=False,
                    axis_config={"color": interpolate_color(BACKGROUND_COLOR, WHITE, 0.5)},).to_edge(RIGHT)
        graphs += grid
        graphs += Text(r"Rel. feature imporance to objectives (%)", font_size=very_small_size).rotate(PI/2).next_to(grid, LEFT)
        def rect_verts(grid, x, y, w, h):
            return [grid.c2p(x, y), grid.c2p(x+w, y), grid.c2p(x+w, y+h), grid.c2p(x, y+h)]
        bar1 = [Polygon(*rect_verts(grid, e['x'], 2, 0.5, e['w']), color=GRAPH_COLOR, fill_opacity=0.5) for e in [{'w': 4, 'x': 2.5}, {'w': 3, 'x': 5.5}, {'w': 1, 'x': 8.5}]]
        bar2 = [Polygon(*rect_verts(grid, e['x'], 2, 0.5, e['w']), color=GREEN, fill_opacity=0.5) for e in [{'w': 1, 'x': 3.1}, {'w': 5, 'x': 6.1}, {'w': 2, 'x': 9.1}]]
        tts = [Text(f"Param. {i}", font_size=very_small_size).move_to(grid.c2p(3+3*i, 1.5)) for i in range(3)]
        graphs.add(*bar1, *bar2, *tts)
        self.play(DrawBorderThenFill(graphs))

        self.next_slide()
        keep_only_objects(self, layout)

        pros = Text("- Better Bayesian Optimization algorithms for CFD?", font_size=mid_size).next_to(title, DOWN*2).align_to(title, LEFT)
        self.play(Create(pros))

        items = [
            "Sparse Axis-Aligned Subspace Bayesian Optimization (SAASBO) for high-dimensionality",
            "More surrogate models: Random Forests, etc.",
        ]
        last = self.itemize(items, pros, 1.5, True,
            t2w={f"1{ITEM_ICON}": BOLD, f"2{ITEM_ICON}": BOLD, f"3{ITEM_ICON}": BOLD, f"4{ITEM_ICON}": BOLD},
            t2c={f"1{ITEM_ICON}": GREEN, f"2{ITEM_ICON}": GREEN, f"3{ITEM_ICON}": GREEN, f"4{ITEM_ICON}": GREEN})

        self.next_slide()

        pros = Text("- Settings trade-offs", font_size=mid_size).next_to(last, DOWN*2).align_to(last, LEFT)
        self.play(Create(pros))

        items = [
            "Number of initial samples -> Initial acquisition function",
            "Number of parallel trials -> Total number of trials",
        ]
        last = self.itemize(items, pros, 1.5, True,
            t2w={f"1{ITEM_ICON}": BOLD, f"2{ITEM_ICON}": BOLD, f"3{ITEM_ICON}": BOLD, f"4{ITEM_ICON}": BOLD},
            t2c={f"1{ITEM_ICON}": GREEN, f"2{ITEM_ICON}": GREEN, f"3{ITEM_ICON}": GREEN, f"4{ITEM_ICON}": GREEN})

        self.next_slide()
        keep_only_objects(self, layout)

        ## SECTION OPENFOAM

        t4 = Text(f"Bayes. Opt. for OpenFOAM", font_size=big_size).to_edge(UP+LEFT)
        self.play(Transform(title, t4))

        pros = Text("- FoamBO package", font_size=mid_size).next_to(title, DOWN*2).align_to(title, LEFT)
        self.play(Create(pros))

        items = [
            "Based on ax-platform",
            "No-code, configuration based",
            "Automatically picks most settings",
            "Run trials locally or on SLURM clusters",
            "Online visualization, you choose how!",
        ]
        last = self.itemize(items, pros, 1.5, True,
            t2w={f"1{ITEM_ICON}": BOLD, f"2{ITEM_ICON}": BOLD, f"3{ITEM_ICON}": BOLD, f"4{ITEM_ICON}": BOLD, f"5{ITEM_ICON}": BOLD},
            t2c={f"1{ITEM_ICON}": GREEN, f"2{ITEM_ICON}": GREEN, f"3{ITEM_ICON}": GREEN, f"4{ITEM_ICON}": GREEN, f"5{ITEM_ICON}": GREEN})

        self.next_slide()

        cons = Text("- Current weaknesses (v0.0.2)", font_size=mid_size).next_to(last, DOWN*2).align_to(last, LEFT)
        self.play(Create(cons))

        items = [
            "Dependent parameters? Meh...",
            "Not-so-reliable software -> Penalize failures for now",
            "No use of constraints"
        ]
        last = self.itemize(items, cons, 1.5, True,
            t2w={f"1{ITEM_ICON}": BOLD, f"2{ITEM_ICON}": BOLD, f"3{ITEM_ICON}": BOLD, f"4{ITEM_ICON}": BOLD, "the Most": BOLD},
            t2c={f"1{ITEM_ICON}": RED, f"2{ITEM_ICON}": RED, f"3{ITEM_ICON}": RED, f"4{ITEM_ICON}": RED})

        self.next_slide()
        keep_only_objects(self, layout)

        box0 = VGroup()
        rc = Rectangle(width=2, height=0.5, color=WHITE)
        tt = Text("FoamBO", font_size=mid_size).next_to(rc, IN, buff=0.2)
        box0.add(rc, tt).shift(UP*2)
        self.play(GrowFromEdge(box0, UP))

        box1 = VGroup()
        rc = Rectangle(width=3, height=0.5, color=WHITE)
        tt = Text("OpenFOAM case", font_size=mid_size).next_to(rc, IN, buff=0.2)
        box1.add(rc, tt).next_to(box0, 2*RIGHT+DOWN)
        self.play(GrowFromEdge(box1, UP))

        case1 = [
            Text("Allrun", font_size=mid_size).next_to(box1, DOWN, buff=0.2),
            Text("Post-processing scripts", font_size=mid_size).next_to(box1, DOWN, buff=0.6)
        ]
        self.play(AnimationGroup(*[FadeIn(c) for c in case1]))

        self.next_slide()
        box2 = VGroup()
        rc = Rectangle(width=3, height=0.5, color=WHITE)
        tt = Text("config.yaml", font_size=mid_size).next_to(rc, IN, buff=0.2)
        box2.add(rc, tt).next_to(box0, LEFT*2+DOWN)
        self.play(GrowFromEdge(box2, UP))

        case2 = [
            Text("Optimization Settings", font_size=mid_size).next_to(box2, DOWN, buff=0.2),
            Text("Objective definition", font_size=mid_size).next_to(box2, DOWN, buff=0.6),
            Text("Parameters definition", font_size=mid_size).next_to(box2, DOWN, buff=1.0),
            Text("Parameter substitution", font_size=mid_size).next_to(box2, DOWN, buff=1.4),
        ]
        self.play(AnimationGroup(*[FadeIn(c) for c in case2]))

        self.next_slide()
        box3 = VGroup()
        rc = Rectangle(width=3, height=0.5, color=GRAPH_COLOR)
        tt = Text("local/SLURM", font_size=mid_size, color=GRAPH_COLOR).next_to(rc, IN, buff=0.2)
        box3.add(rc, tt).next_to(box0, 11*DOWN)
        self.play(
            AnimationGroup(*[c.animate.set_color(GRAPH_COLOR) for c in [case1[0], case1[1], case2[1]]]),
            GrowFromEdge(box3, DOWN)
        )

        self.next_slide()
        keep_only_objects(self, layout)

        ## SECTION DEMO

        t5 = Text(f"Sandia Flame D case", font_size=big_size).to_edge(UP+LEFT)
        self.play(Transform(title, t5))

        flame = ImageMobject("./images/minipilot.png").scale(0.6).to_edge(RIGHT).shift(LEFT)
        self.play(FadeIn(flame))

        pros = Text("- Can we improve on the standard OpenFOAM tutorial?", font_size=mid_size).next_to(title, DOWN*2).align_to(title, LEFT)
        self.play(Create(pros))

        items = [
            "Predict CH4, CO2, T and U",
            "Watching Execution time",
            "And continuity errors!",
        ]
        last = self.itemize(items, pros, 1.5, True,
            t2w={f"1{ITEM_ICON}": BOLD, f"2{ITEM_ICON}": BOLD, f"3{ITEM_ICON}": BOLD, f"4{ITEM_ICON}": BOLD},
            t2c={f"1{ITEM_ICON}": GREEN, f"2{ITEM_ICON}": GREEN, f"3{ITEM_ICON}": GREEN, f"4{ITEM_ICON}": GREEN})
        self.next_slide()

        cons = Text("- Searching for a better params set:", font_size=mid_size).next_to(last, DOWN*2).align_to(last, LEFT)
        self.play(Create(cons))

        items = [
            "Turbulence: laminar, kEpsilon, LaunderSharmaKE",
            "Chemistry Mechansim: GRI3, DRM22",
            "Chemistry Type: ODE, EulerImplicit",
            #"Combustion: EDC, EDM, none, laminar",
            "Combustion: EDC, none, laminar",
            "Mesh resolution: [2->7] (default: 5)",
            #"mixture: reactingMixture, singleStepReactingMixure",
        ]
        last = self.itemize(items, cons, 1.5, True,
            t2w={f"1{ITEM_ICON}": BOLD, f"2{ITEM_ICON}": BOLD, f"3{ITEM_ICON}": BOLD, f"4{ITEM_ICON}": BOLD, f"5{ITEM_ICON}": BOLD, f"6{ITEM_ICON}": BOLD},
            t2c={f"1{ITEM_ICON}": RED, f"2{ITEM_ICON}": RED, f"3{ITEM_ICON}": RED, f"4{ITEM_ICON}": RED, f"5{ITEM_ICON}": RED,  f"6{ITEM_ICON}": RED})

        self.next_slide()
        keep_only_objects(self, layout)

        pros = Text("- Configure objectives", font_size=mid_size).next_to(title, DOWN*2).align_to(title, LEFT)
        self.play(Create(pros))

        items = [
            ["TemperatureMSE", ""],
            ['mode', 'shell'],
            ['command', "pvpython postprocess.py --T --decomposed"],
            ['threshold', '5e-2'],
            ['minimize', 'True']
        ]
        indents = [0, 0.5, 0.5, 0.5, 0.5]
        last = self.hi_yaml(items, indents, pros, 1)

        self.next_slide()

        cons = Text("- Configure Parameters", font_size=mid_size).next_to(last, DOWN*2).align_to(last, LEFT).shift(0.5*LEFT)
        self.play(Create(cons))

        items = [
            ["chemistryMechanism", ""],
            ['type', 'choice'],
            ['value_type', 'str'],
            ['values', '[GRI3, DRM22]'],
            ['is_ordered', 'True']
        ]
        indents = [0, 0.5, 0.5, 0.5, 0.5]
        last = self.hi_yaml(items, indents, cons, 1)

        self.next_slide()

        cons = Text("- Substitue Parameters", font_size=mid_size).next_to(cons, RIGHT*8)
        self.play(Create(cons))

        items = [
            ["file_copies", ""],
            ["chemistryMechanism", ""],
            ["template", "/chemkin/mechanismProperties"]
        ]
        indents = [0, 0.5, 1]
        last = self.hi_yaml(items, indents, cons, 1)

        self.next_slide()
        keep_only_objects(self, layout)

        ### SECTION OBJECTIVES

        flame = ImageMobject("./images/flame.png")#.to_edge(UP+RIGHT)
        self.play(FadeIn(flame))

        self.next_slide()
        keep_only_objects(self, layout)
        df1 = pd.read_csv(f"../{CASE_NAME}_report.csv")
        df = df1[~df1.isin([1.0]).any(axis=1)]

        maxMSE = df[['CO2MSE', 'CH4MSE', 'TemperatureMSE', 'VelocityMSE']].max().max()

        graphs = VGroup()
        grid = Axes(x_range=[0, len(df1), 1], y_range=[0, maxMSE, maxMSE/5.0],
                    x_length=10, y_length=5, tips=False,
                    axis_config={"color": interpolate_color(BACKGROUND_COLOR, WHITE, 0.5)},).to_edge(RIGHT)
        for i in range(0, len(df1), 5):
            graphs += Text(f'{i}', font_size=very_small_size).rotate(PI/4).move_to(grid.c2p(i, 0)).shift(0.25*DOWN)
        for i in np.linspace(0, maxMSE, 6):
            graphs += Text(f'{i:.2e}', font_size=very_small_size).rotate(PI/4).move_to(grid.c2p(0, i)).shift(0.5*LEFT)
        graphs += grid
        graphs += Text(r"MSE in T, U, CH4, and CO2", font_size=very_small_size).rotate(PI/2).next_to(grid, LEFT).shift(0.7*LEFT)
        graphs += Text(r"Trial Index", font_size=very_small_size).next_to(grid, DOWN).shift(0.2*DOWN)
        self.play(DrawBorderThenFill(graphs))

        def get_stroke(gen, default):
            return  default if gen == 'GPEI' else WHITE

        def get_stroke_width(gen, default):
            return  default if gen == 'GPEI' else 1.5

        self.next_slide()
        t_dots = [Dot(color=DOT_COLOR,
                      stroke_width=get_stroke_width(df['generation_method'][i], 0),
                      stroke_color=get_stroke(df['generation_method'][i], DOT_COLOR)
            ).move_to(grid.c2p(i, j))
            for i,j in zip(df['trial_index'], df['TemperatureMSE'])]
        tt = Text(r"Temperature", color=DOT_COLOR, font_size=very_small_size).next_to(grid, UP)
        tc = VGroup()
        tc1 = Dot(color=DOT_COLOR,
                  stroke_width=get_stroke_width('SOBOL', 0),
                  stroke_color=get_stroke('SOBOL', DOT_COLOR),
                  fill_opacity=0.0
        )
        tc2= Text(r"SOBOL", color=WHITE, font_size=very_small_size).next_to(tc1, RIGHT)
        tc.add(tc1,tc2)
        tc.next_to(tt, RIGHT)
        self.play(AnimationGroup(*[Create(d) for d in t_dots]), Create(tt), Create(tc))

        self.next_slide()
        u_dots = [Dot(color=GRAPH_COLOR,
                      stroke_width=get_stroke_width(df['generation_method'][i], 0),
                      stroke_color=get_stroke(df['generation_method'][i], GRAPH_COLOR)
            ).move_to(grid.c2p(i, j))
            for i,j in zip(df['trial_index'], df['VelocityMSE'])]
        tt = Text(r"Velocity", color=GRAPH_COLOR, font_size=very_small_size).next_to(tt, LEFT)
        self.play(AnimationGroup(*[Create(d) for d in u_dots]), Create(tt))

        self.next_slide()
        co2_dots = [Dot(color=GREEN,
                      stroke_width=get_stroke_width(df['generation_method'][i], 0),
                      stroke_color=get_stroke(df['generation_method'][i], GREEN)
            ).move_to(grid.c2p(i, j))
            for i,j in zip(df['trial_index'], df['CO2MSE'])]
        tt = Text(r"CO2", color=GREEN, font_size=very_small_size).next_to(tt, LEFT)
        self.play(AnimationGroup(*[Create(d) for d in co2_dots]), Create(tt))

        self.next_slide()
        ch4_dots = [Dot(color=ORANGE,
                      stroke_width=get_stroke_width(df['generation_method'][i], 0),
                      stroke_color=get_stroke(df['generation_method'][i], ORANGE)
            ).move_to(grid.c2p(i, j))
            for i,j in zip(df['trial_index'], df['CH4MSE'])]
        tt = Text(r"CH4", color=ORANGE, font_size=very_small_size).next_to(tt, LEFT)
        self.play(AnimationGroup(*[Create(d) for d in ch4_dots]), Create(tt))

        self.next_slide()
        keep_only_objects(self, layout)

        graphs = VGroup()
        grid = Axes(x_range=[0, len(df1), 1], y_range=[0, df['ExecutionTime'].max(), df['ExecutionTime'].max()/5.0],
                    x_length=10, y_length=5, tips=False,
                    axis_config={"color": interpolate_color(BACKGROUND_COLOR, WHITE, 0.5)},).to_edge(RIGHT)
        graphs += grid
        for i in range(0, len(df1), 5):
            graphs += Text(f'{i}', font_size=very_small_size).rotate(PI/4).move_to(grid.c2p(i, 0)).shift(0.25*DOWN)
        for i in np.linspace(0, df['ExecutionTime'].max(), 6):
            graphs += Text(f'{i:.2f}', font_size=very_small_size).rotate(PI/4).move_to(grid.c2p(0, i)).shift(0.5*LEFT)
        graphs += Text(r"Execution Time", font_size=very_small_size).rotate(PI/2).next_to(grid, LEFT).shift(0.5*LEFT)
        graphs += Text(r"Trial Index", font_size=very_small_size).next_to(grid, DOWN).shift(0.2*DOWN)
        self.play(DrawBorderThenFill(graphs))
        time_dots = [Dot(color=DOT_COLOR,
                      stroke_width=get_stroke_width(df['generation_method'][i], 0),
                      stroke_color=get_stroke(df['generation_method'][i], DOT_COLOR)
            ).move_to(grid.c2p(i, j))
            for i,j in zip(df['trial_index'], df['ExecutionTime'])]
        self.play(AnimationGroup(*[Create(d) for d in time_dots]))

        self.next_slide()
        keep_only_objects(self, layout)

        ### SECTION FLAME PARETO

        t6 = Text(f"Sandia Flame D case: Pareto front", font_size=big_size).to_edge(UP+LEFT)
        self.play(Transform(title, t6))

        df1 = pd.read_csv(f"../{CASE_NAME}_frontier_report.csv")
        df = df1[~df1.isin([1.0]).any(axis=1)]

        graphs = VGroup()
        grid = Axes(x_range=[0, df['ExecutionTime'].max(), df['ExecutionTime'].max()/5],
                    y_range=[0, df['CO2MSE'].max(), df['CO2MSE'].max()/5.0],
                    x_length=10, y_length=5, tips=False,
                    axis_config={"color": interpolate_color(BACKGROUND_COLOR, WHITE, 0.5)},).to_edge(RIGHT)
        graphs += grid
        for i in np.linspace(0, df['ExecutionTime'].max(), 6):
            graphs += Text(f'{i:.0f}', font_size=very_small_size).rotate(PI/4).move_to(grid.c2p(i, 0)).shift(0.25*DOWN)
        for i in np.linspace(0, df['CO2MSE'].max(), 6):
            graphs += Text(f'{i:.2e}', font_size=very_small_size).rotate(PI/4).move_to(grid.c2p(0, i)).shift(0.5*LEFT)
        graphs += Text(r"MSE in CO2", font_size=very_small_size).rotate(PI/2).next_to(grid, LEFT).shift(0.5*LEFT)
        graphs += Text(r"Execution Time", font_size=very_small_size).next_to(grid, DOWN).shift(0.2*DOWN)
        self.play(DrawBorderThenFill(graphs))
        front_dots = [Dot(color=DOT_COLOR, radius=0.03).move_to(grid.c2p(i, j))
            for i,j in zip(df['ExecutionTime'], df['CO2MSE'])]
        xlines = [Line(start=grid.c2p(i-j, k), end=grid.c2p(i+j, k), color=DOT_COLOR)
            for i,j,k in zip(df['ExecutionTime'], df['ExecutionTime_sems'], df['CO2MSE'])]
        ylines = [Line(start=grid.c2p(k, i-j), end=grid.c2p(k, i+j), color=DOT_COLOR)
            for i,j,k in zip(df['CO2MSE'], df['CO2MSE_sems'], df['ExecutionTime'])]
        params = [
            df[['turbulenceModel', 'chemistryMechanism', 'chemistryType', 'combustionModel', 'meshResolution']].loc[i]
            for i in [2, 15]
        ]
        print(params)
        self.play(AnimationGroup(*[Create(d) for d in front_dots]))
        self.play(AnimationGroup(*[Create(d) for d in xlines], *[Create(d) for d in ylines]))
        txts = [[Text(f'{i}: {e[i]}', font_size=very_small_size) for i in e.index] for e in params]
        self.play(Create(VGroup(*txts[0]).arrange(0.4*DOWN, center=False, aligned_edge=RIGHT).next_to(front_dots[2], RIGHT).shift(0.5*DOWN)))
        self.play(Create(VGroup(*txts[1]).arrange(0.4*DOWN, center=False, aligned_edge=LEFT).next_to(front_dots[15], LEFT).shift(0.5*UP)))

        self.play(Create(Dot(color=DOT_COLOR).move_to(100*LEFT)))

        self.next_slide()
        keep_only_objects(self, layout)

        ### SECTION FLAME REL FEATURES

        df = pd.read_csv(f"../{CASE_NAME}_feature_importance_report.csv")
        maxImportance = df.max()[~df.columns.isin(['objective'])].max()*100

        t7 = Text(f"Sandia Flame D case: Rel. importance", font_size=big_size).to_edge(UP+LEFT)
        self.play(Transform(title, t7))

        graphs = VGroup()
        grid = Axes(x_range=[0, len(df.columns)-1, 1],
                    y_range=[0, 50, 10],
                    x_length=10, y_length=5, tips=False,
                    axis_config={"color": interpolate_color(BACKGROUND_COLOR, WHITE, 0.5)},).to_edge(RIGHT)
        graphs += grid
        for i in range(1, len(df.columns)):
            graphs += Text(f'{df.columns[i]}', font_size=very_small_size).move_to(grid.c2p(i-0.6, 0)).shift(0.25*DOWN)
        for i in np.linspace(0, 50, 6):
            graphs += Text(f'{i:.1f}%', font_size=very_small_size).rotate(PI/4).move_to(grid.c2p(0, i)).shift(0.5*LEFT)
        graphs += Text(r"Relative importance", font_size=very_small_size).rotate(PI/2).next_to(grid, LEFT).shift(0.5*LEFT)
        self.play(DrawBorderThenFill(graphs))


        cols = [DOT_COLOR, GRAPH_COLOR, GREEN, ORANGE, YELLOW, BLUE]
        for i in range(len(df)):
            row = df.loc[i]
            obj = row['objective']
            params = row.index[1:]
            bars = [Polygon(*rect_verts(grid, 0.12*i+j, 0, 0.12, row[params[j]]*100), color=cols[i], fill_opacity=0.5) for j in range(len(params))]
            tt = Text(f"{obj}", color=cols[i], font_size=very_small_size).align_to(grid, LEFT).shift(3*LEFT+i*0.5*UP)
            self.play(AnimationGroup(*[DrawBorderThenFill(b) for b in bars]), Create(tt))
            self.next_slide()

        #### SECTION CONCLUSION

        keep_only_objects(self, layout)

        t8 = Text(f"Key takeaways", font_size=big_size).to_edge(UP+LEFT)
        self.play(Transform(title, t8))

        pros = Text("- Bayesian Optimization", font_size=mid_size).next_to(title, DOWN*2).align_to(title, LEFT)
        self.play(Create(pros))

        items = [
            "Can potentially do better for optimizing expensive black-box obj. funcs.",
            "The surrogate model is useful for further analysis.",
            "The acquisition function plays a crucial role",
            "For CFD -> Multi-objective optimization is the way to go!",
        ]
        last = self.itemize(items, pros, 1.5, True,
            t2w={f"1{ITEM_ICON}": BOLD, f"2{ITEM_ICON}": BOLD, f"3{ITEM_ICON}": BOLD, f"4{ITEM_ICON}": BOLD},
            t2c={f"1{ITEM_ICON}": GREEN, f"2{ITEM_ICON}": GREEN, f"3{ITEM_ICON}": GREEN, f"4{ITEM_ICON}": GREEN})

        self.next_slide()

        cons = Text("- Multi-Obj Opt. for CFD", font_size=mid_size).next_to(last, DOWN*2).align_to(last, LEFT)
        self.play(Create(cons))

        items = [
            "Trade-off between objectives translate into Pareto fronts",
            "Decision making can rely on feature importances",
        ]
        last = self.itemize(items, cons, 1.5, True,
            t2w={f"1{ITEM_ICON}": BOLD, f"2{ITEM_ICON}": BOLD, f"3{ITEM_ICON}": BOLD, f"4{ITEM_ICON}": BOLD},
            t2c={f"1{ITEM_ICON}": GREEN, f"2{ITEM_ICON}": GREEN, f"3{ITEM_ICON}": GREEN, f"4{ITEM_ICON}": GREEN})

        self.wait()
        self.next_slide()  # Waits user to press continue to go to the next slide
        keep_only_objects(self, layout)

        self.play(FadeOut(title))
        thanks = Text("Thank you for your attention!", font_size=big_size)
        self.play(Create(thanks))
