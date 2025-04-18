

function plot_perf_evolution_early(early_perf_filename, measure, env_name, out_filename; scale="linear", axs=nothing, novig=false)
    early_perf = load(early_perf_filename)
    early_scores = early_perf["scores"]
    measures = early_perf["measures"]
    intr_types = novig ? [0.0, 0.25, 0.5, 0.75, 1.0] : early_perf["intr_types"]
    early_n_repet = size(early_scores)[end]
    early_step_size = early_perf["step_size"]
    early_n_steps = size(early_scores)[3]
    early_steps = [s*early_step_size for s in 1:early_n_steps]

    measure_title = Dict(
        "visit_all_states" => "Fraction of unvisited states", 
        "Phat_approximation" => "Model accuracy",
        "state_freq" => "State frequency vs uniform"
    )

    #if axs==nothing
    #    fig, axs = subplots(1, 2, figsize=(5,5), sharey=true)
    #    fig.subplots_adjust(wspace=0.05)
    #end

    for (im, meas) in enumerate(measures)
        if meas != measure
            continue
        end
        for (it, intr_type) in enumerate(intr_types)
            
            early_s = early_scores[im,it,:,:]

            early_mean_score = dropdims(mean(early_s, dims=2), dims=2)
            early_std = dropdims(std(early_s, dims=2), dims=2)
            early_st_err = early_std ./ sqrt(early_n_repet)
            # Quantiles
            #early_q = mapslices(x -> quantile(x, [0.25, 0.75]), early_s; dims=2)

            
            axs.plot(early_steps, early_mean_score, label=intr_type)
            #axs.fill_between(early_steps, early_q[:,1], early_q[:,2], alpha=0.2)
            axs.fill_between(early_steps, early_mean_score .- early_st_err, early_mean_score .+ early_st_err, alpha=0.2)
            axs.set_yscale(scale)

        end
        
    end
end

# setup: env_name, early_steps
function plot_all_early(setups, plot_filename, perf_prefix=""; novig=false, graph_img_folder="")
    measures = ["visit_all_states", "Phat_KL", "state_freq_KL"]#["visit_all_states", "Phat_approximation", "state_freq"]#
    measure_title = Dict(
        "visit_all_states" => "1. State discovery", 
        "Phat_approximation" => "2. Model accuracy",
        "state_freq" => "3. Uniform state visitation",
        "Phat_KL" => "2. Model accuracy (KL)",
        "state_freq_KL" => "3. Uniform state visitation (KL)"
    )
    measure_complement = Dict(
        "visit_all_states" => "% of unvisited states", 
        "Phat_approximation" => latexstring("RMSE(\$\\hat{P}_{s,a}, P_{s,a}\$)"),
        "state_freq" => latexstring("RMSE(\$p(S), Unif(S)\$)"),
        "Phat_KL" => latexstring("KL(\$P_{s,a}, \\hat{P}_{s,a}\$)"),
        "state_freq_KL" => latexstring("KL(\$p(S), Unif(S)\$)")
    )
    env_title = Dict(
        "default" => "Neutral", 
        "sink50" => "Sink",
        "sink100" => "Sink",
        "stoc1" => "Stochastic",
        "sink50-source50-stoc1" => "Mixed",
        "source50" => "Source",
        "source100" => "Source"
    )

    fig = figure(figsize=(20,25))#(length(measures)*4, length(setups)*3))
    handles = nothing
    labels=nothing
    gs = fig.add_gridspec(length(setups), length(measures)+1, wspace=0.3, hspace=0.3)
    #fig, axs = plt.subplots(length(setups), length(measures)*2, figsize=(length(measures)*4, length(setups)*2))
    #fig.subplots_adjust(wspace=0.05)
    #x_axis_early = nothing


    for (is,setup) in enumerate(setups)

        env_name, early_steps = setup

        ax = fig.add_subplot(gs[is,1])
        ax.axis("off")
        if graph_img_folder != ""
            im_file = graph_img_folder * env_name * ".png"
            im = imread(im_file)
            ax.imshow(im)
        end
        ax.text(-0.3, 0.5, env_title[env_name], transform=ax.transAxes, ha="left", va="center", rotation="vertical", fontsize=20)
        if is==1
            ax.text(0.5, 1.5, "Examples", transform=ax.transAxes, ha="center", fontsize=22, color="black", fontweight="bold")
        end

        early_perf_filename = perf_prefix*env_name*"_"* string(early_steps)*".jld"
        
        for (im,meas) in enumerate(measures)
            sub_gs = gs[is, im+1].subgridspec(1,1, wspace=0.1)
            sub_axs = sub_gs.subplots(sharey=true)
            
            #if x_axis_early == nothing
            #    x_axis_early = sub_axs[1]
            #else
                #sub_axs[1].sharex(x_axis_early)
                #sub_axs[2].sharex(x_axis_late)
                
            #end

            #plot_filename = "data/plots/partial_perfs/opti_by_measure/opti8/one_by_one/"*env_name*"_"*meas*"_early-late_10-90.png"
            sub_axs.set_xlim(0,2000)
            scale = meas=="state_freq" ? "linear" : "linear"
            plot_perf_evolution_early(early_perf_filename, meas, env_title[env_name], "", scale=scale, axs=sub_axs, novig=novig)
            handles, labels = sub_axs.get_legend_handles_labels()
            ax = fig.add_subplot(gs[is,im+1])
            ax.set_xlim(0,2)
            ax.patch.set_alpha(0)
            ax.axis("off")
            # plot the vertical line
            if is == 1
                ax.text(0.5, 1.3, measure_title[meas], transform=ax.transAxes, ha="center", fontsize=20, color="black")
                ax.text(0.5, 1.15, measure_complement[meas], transform=ax.transAxes, ha="center", fontsize=20, color="grey")
                if im == 2
                    ax.text(0.5, 1.5, "Performance measures", transform=ax.transAxes, ha="center", fontsize=22, fontweight="bold")
                    ax.text(0.05, 0.5, "Environment classes", transform=fig.transFigure, ha="left", va="center", rotation="vertical", fontsize=22, fontweight="bold")
                end
            elseif is == length(setups)
                ax.text(0.5, -0.25, "Steps", transform=ax.transAxes, ha="center", fontsize=20)
            end

            #ax.set_title(measure_title[meas])
        end
    end
    intr_types = ["Novelty", "Surprise", "Information gain", "Naive empowerment", "MOP", "SPIE", "Random"]

    #PyPlot.tight_layout()
    
    
    leg = PyPlot.legend(handles, intr_types, bbox_to_anchor=(0.5, 0.02), loc="lower center", bbox_transform=fig.transFigure, ncol=5, fontsize=20)
    for legobj in leg.legend_handles
        legobj.set_linewidth(5.0)
    end

    PyPlot.tight_layout()
    PyPlot.savefig(plot_filename, dpi=300, bbox_inches="tight", pad_inches=0.5)
    PyPlot.close()
end


function plot_perf_across_envs(perf_filenames, out_filename, xticks, xlabel; scale="auto", 
    intr_names=["Novelty", "Surprise", "Information gain", "Naive empowerment", "MOP", "SPIE", "Random"]
) 
data_perf = load(perf_filenames[1])
scores = data_perf["scores"]
n_repet = size(scores)[end]
mean_steps = dropdims(mean(scores, dims=3), dims=3)
mean_scores = dropdims(mean(mean_steps, dims=3), dims=3)
std_scores = dropdims(std(mean_steps, dims=3), dims=3)
measures = data_perf["measures"]
intr_types = data_perf["intr_types"]
#intr_names = intr_types
#n_repet = size(scores)[end]
#step_size = data_perf["step_size"]
#n_steps = size(scores)[3]
#steps = [s*step_size for s in 1:n_steps]

for filename in perf_filenames[2:end]
    data_perf = load(filename)
    mean_steps = dropdims(mean(data_perf["scores"], dims=3), dims=3)
    mean_s = dropdims(mean(mean_steps, dims=3), dims=3)
    std_s = dropdims(std(mean_steps, dims=3), dims=3)
    #mean_s = dropdims(mean(data_perf["scores"], dims=(3,4)), dims=(3,4)) # shape: measures, intr_types
    mean_scores = cat(mean_scores, mean_s, dims=3) # shape: measures, intr_types, n_files
    std_scores = cat(std_scores, std_s, dims=3) # shape: measures, intr_types, n_files
end
#mean_scores = permutedims(mean_scores, [1,3,2])
#min_per_envmeas = dropdims(minimum(mean_scores, dims=3), dims=3)
#max_per_envmeas = dropdims(maximum(mean_scores, dims=3), dims=3)

#norm_scores = (mean_scores .- min_per_envmeas) ./ (max_per_envmeas - min_per_envmeas)

measure_title = Dict(
    "visit_all_states" => "State discovery", 
    "Phat_approximation" => "Model accuracy",
    "state_freq" => "Uniform state visitation",
    "Phat_KL" => "Model accuracy (KL)",
    "state_freq_KL" => "Uniform state visitation (KL)"
)
measure_complement = Dict(
    "visit_all_states" => "% of unvisited states", 
    "Phat_approximation" => latexstring("RMSE(\$\\hat{P}_{s,a}, P_{s,a}\$)"),
    "state_freq" => latexstring("RMSE(\$p(S), Unif(S)\$)"),
    "Phat_KL" => latexstring("KL(\$P_{s,a}, \\hat{P}_{s,a}\$)"),
    "state_freq_KL" => latexstring("KL(\$p(S), Unif(S)\$)")
)
colors = Dict(
    "novelty" => "tab:blue",
    "surprise" => "tab:orange",
    "information_gain" => "tab:green",
    "empowerment" => "tab:red",
    "MOP" => "tab:purple",
    "SP" => "tab:brown",
    "random" => "tab:pink",
    "novelty_eps" => "tab:blue",
    "novelty_1/c" => "tab:orange",
    "novelty_1/sqrt" => "tab:green"
)

fig, axs = subplots(1, length(measures), figsize=(19,6))

for (im, meas) in enumerate(measures)
    axs[im].set_xlim(xticks[1], xticks[end])
    
    for (it, intr_type) in enumerate(intr_types)
        s = mean_scores[im,it,:]
        stdev = std_scores[im,it,:]
        sterr = stdev ./ sqrt(n_repet)
        #mean_score = dropdims(mean(s, dims=2), dims=2)
        #sample_dev = sqrt.(
        #    dropdims(
        #        sum((s.-mean_score).^2, dims=2), 
        #        dims=2)
        #    ./(n_repet-1)
        #    )
        #err_bar = 1.96 .* sample_dev ./ sqrt(n_repet)
        # Quantiles
        #q = mapslices(x -> quantile(x, [0.1, 0.9]), s; dims=2)
        
        axs[im].plot(xticks, s, label=intr_names[it], c=colors[intr_type], linewidth=2)
        #axs[im].fill_between(steps, q[:,1], q[:,2], alpha=0.2)
        axs[im].fill_between(xticks, s .- sterr, s.+sterr, alpha=0.2)
        sca = scale
        if scale=="auto"
            sca = meas == "state_freq" || meas == "Phat_approximation" ? "log" : "linear"
        end 
        axs[im].set_yscale(sca)

        #axs[im].fill_between(steps, mean_score.+err_bar, mean_score.-err_bar, alpha=0.2)
        #axs[im].errorbar(steps, mean_score, yerr=sample_dev, marker="o", linestyle="none")
    end
    
    #axs[im].set_title("Measure: "*measure_title[meas])
    axs[im].text(0.5, 1.15, measure_title[meas], transform=axs[im].transAxes, ha="center", fontsize=20, color="black")
    axs[im].text(0.5, 1.06, measure_complement[meas], transform=axs[im].transAxes, ha="center", fontsize=20, color="grey")
    axs[im].set_xlabel(xlabel)
end
axs[1].text(-0.05, 1.1, "B", transform=axs[1].transAxes, ha="center", fontsize=22, fontweight="bold")
#suptitle("Performance evolution")
axs[2].legend()
println(out_filename)
tight_layout()
PyPlot.savefig(out_filename, dpi=300)
PyPlot.close()
end



function plot_avg_normalized(perf_filenames, big_plot_filename, avg_plot_filename, env_names, intr_names; n_steps=100, normalization_avgs=nothing, novig=false, savedata_file=nothing, nov_ig_data_file=nothing, use_last=false, extr_max=3000)
    colors = novig ? ["tab:blue", "lightsteelblue", "lightgrey", "darkseagreen", "tab:green"] : ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:pink"]

    measure_title = Dict(
        "visit_all_states" => "State discovery", 
        "Phat_approximation" => "Model accuracy",
        "state_freq" => "Uniform state visitation",
        "Phat_KL" => "Model accuracy (KL)",
        "state_freq_KL" => "Uniform state visitation (KL)",
        "extr_start" => "Persistent extrinsic reward",
        "extr_2000_known" => "Announced late extrinsic reward",
        "extr_2000_unknown" => "Unannounced late extrinsic reward",
    )

    extr_meas_start = Dict(
        "extr_start" => 0,
        "extr_2000_known" => 2000,
        "extr_2000_unknown" => 2000
    )

    d = load(perf_filenames[1])
    measures = d["measures"]
    intr_types = novig ? [0.0, 0.25, 0.5, 0.75, 1.0] : d["intr_types"]
    if length(intr_types) == 8
        # remove last element
        intr_types = intr_types[1:end-1]
    end
    n_env = length(perf_filenames)
    n_meas = length(measures)
    n_intr = length(intr_types)

    norm_scores = nothing
    if isfile(savedata_file)
        d = load(savedata_file)
        norm_scores = d["norm_scores"]
        if nov_ig_data_file != nothing && isfile(nov_ig_data_file)
            norm_scores2 = load(nov_ig_data_file)["norm_scores"]
            norm_scores[:,:,5] .= norm_scores2[:,:,1]
            norm_scores[:,:,1] .= norm_scores2[:,:,3]
        end
    else
        avgs = zeros(n_env, n_meas, n_intr)

        for (ip,perf_file) in enumerate(perf_filenames)
            d = load(perf_file)
            scores = d["scores"][:,:,1:n_steps, :]
            if use_last
                scores = scores[:,:,end,:] # shape (n_meas, n_intr+1 (extrinsic), n_repet)
                # remove random row in second dimension
                if !novig
                    #scores = cat(scores[:,1:6,:], scores[:,8:end,:], dims=2)
                    scores = scores[:,1:end-1,:]
                end

                # replace -1 with extr_max
                scores[scores .== -1] .= extr_max
                # subtract scores with extr_meas_start
                for (im, meas) in enumerate(measures)
                    scores[im, :, :] .-= extr_meas_start[meas]
                end
                scores[scores .> 1000] .= 1000 # max is 1000
                avg = dropdims(mean(scores, dims=3), dims=3)
                avgs[ip,:,:] = avg

            else
                avg = dropdims(mean(scores, dims=(3,4)), dims=(3,4))
                avgs[ip,:,:] = avg
            end
        end    

        if normalization_avgs == nothing
            # Which scores to use for normalization
            normalization_avgs = avgs
        end
        min_per_envmeas = dropdims(minimum(normalization_avgs, dims=3), dims=3)
        max_per_envmeas = dropdims(maximum(normalization_avgs, dims=3), dims=3)

        norm_scores = (avgs .- min_per_envmeas) ./ (max_per_envmeas - min_per_envmeas)
        save(savedata_file, "norm_scores", norm_scores)
    end

    if novig
        norm_scores = reverse(norm_scores, dims=3)
    end

    avg_over_envs = dropdims(mean(norm_scores, dims=1), dims=1)
    std_over_envs = dropdims(std(norm_scores, dims=1), dims=1)
    fig, axs = plt.subplots(1,n_meas, figsize=(19,6))

    width = 1/(n_intr+1)
    for (im,meas) in enumerate(measures)
        for (it, intr_type) in enumerate(intr_types)
            scat = axs[im].scatter([1+width*it+rand(Uniform(-width/6, width/6)) for i in 1:n_env], norm_scores[:,im,it], color="black", s=20, alpha=0.2, label="_nolegend_", zorder=3)
            axs[im].bar(1+width*it, avg_over_envs[im, it], width, label=intr_type, alpha=1.0, zorder=2, color=colors[it])#, yerr=std_over_envs[im,it], capsize=3.0)#, yerr=sd[im,it], capsize=3.0, alpha=alpha)
        end
        axs[im].set_xticks([1+width*it for it in 1:n_intr])
        axs[im].set_xticklabels(intr_names, fontsize=18)#, rotation=45, ha="right")
        axs[im].set_title(measure_title[meas], fontsize=20, pad=20)
        axs[im].set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        if im==1
            axs[im].set_yticklabels(["best agent - 0.0", 0.2, 0.4, 0.6, 0.8, "worst agent - 1.0"], fontsize=18)#, rotation=45, ha="right")
        else
            axs[im].set_yticklabels([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=18)
        end
        axs[im].set_ylim(top=1.0)

        axs[im].set_xlim(1+width/2, 1+width*n_intr+width/2)
        novig && axs[im].set_xlabel(latexstring("\$\\alpha \$"))
        im == 1 && (axs[im].set_ylabel("Normalized score"))
    end

    axs[1].text(-0.05, 1.1, "B", transform=axs[1].transAxes, ha="center", fontsize=22, fontweight="bold")

    tight_layout()
    savefig(avg_plot_filename, dpi=300)
    close()
    return normalization_avgs
end


# Function to calculate the percentage of each digit in a window
function calculate_percentages(window)
    counts = countmap(window)
    total = length(window)
    return [(get(counts, i, 0) / total) for i in 1:5]
end

function stackplot_1ax(ax, time_spent, window_size, colors)
    # Calculate percentages for each sliding window
    n_envs, n_steps = size(time_spent)
    plot_len = n_steps - window_size + 1
    percentages = Dict(i => Vector{Float64}(undef, plot_len) for i in 1:5)


    for i in 1:plot_len
        windows_sum = zeros(Int64, 5) # sum over envs for each window
        for ie in 1:n_envs
            window = time_spent[ie, i:i+window_size-1]
            percentages_window = calculate_percentages(window)
            windows_sum = windows_sum .+ percentages_window
        end

        for j in 1:5
            percentages[j][i] = windows_sum[j] / n_envs # average over envs
        end
    end
    ax.set_xlim(1,plot_len)
    ax.stackplot(1:plot_len, percentages[1], percentages[2], percentages[3], percentages[4], percentages[5], colors = colors)
end

function time_spent_stackplot_full(env_name, intr_types, out_filename; window_size=50, path="", unknown_folder="", known_folder="/", window_sizes=[1000,9999], graph_img_file="")
    colors = PyPlot.cm.Accent.colors#["indianred", "mediumseagreen", "gold", "cornflowerblue", "lightgray"] #(0.93, 0.93, 0.0) is the matplotlib equivalent of yellow2 Julia color
    regions = ["Sink room", "Source room", "Stochastic room", "Neutral room", "Corridor"]
    intr_names = ["Novelty", "Surprise", "Information gain", "Naive empowerment", "MOP", "SPIE", "Random"]

    fig = figure(figsize=(19,9))
    x1len = 4
    x2len = 2
    xpad = 1

    xlen = 4*(x1len + x2len) + 3*xpad

    for (it, intr_type) in enumerate(intr_types)
        i,j = Int(floor((it-1) / 4)), (it-1) % 4

        time_spent = load(path*unknown_folder*env_name*"_"*intr_type*"_βopti.jld")["time_spent"]
        ax1 = subplot2grid((2, xlen), (i, j*(x1len+x2len+xpad)), colspan=x1len)
        stackplot_1ax(ax1, time_spent, window_sizes[1], colors)
        ax1."spines".get("right").set_visible(false)
        ax1."spines".get("top").set_visible(false)
        ax1.set_xlabel("Steps")
        ax1.set_title("During\nlearning", y=0.95, fontsize=14)

        ax1.text(0.75,1.2, intr_names[it], transform=ax1.transAxes, ha="center", fontsize=22, fontweight="bold")

        time_spent = load(path*known_folder*env_name*"_"*intr_type*"_βopti.jld")["time_spent"]
        ax2 = subplot2grid((2, xlen), (i, j*(x1len+x2len+xpad)+x1len), colspan=x2len)
        stackplot_1ax(ax2, time_spent, window_sizes[2], colors)
        ax2.set_yticks([])
        ax2.set_xticks([1.5])
        ax2.set_xticklabels(["Average over\n10000 steps"])

        ax2."spines".get("right").set_visible(false)
        ax2."spines".get("left").set_visible(false)
        ax2."spines".get("top").set_visible(false)
        ax2.set_title("After ideal\nlearning", y=0.95, fontsize=14)

        if intr_names[it]=="Random"
            for i in 1:5
                ax1.bar([i], [0], color=colors[i], label=regions[i])
            end
        end
        

    end
    # Only for the legend
    ax = subplot2grid((2, xlen), (1, 3*(x1len+x2len+xpad)), colspan=x1len+x2len)
    for i in 1:5
        ax.bar([i], [0], color=colors[i], label=regions[i])
    end

    ax.axis("off")
    im_file = graph_img_file
    im = imread(im_file)
    im = ax.imshow(im, extent=[0.2, 1.05, 0.0, 1.2], transform=ax.transAxes)
    im.set_clip_on(false)

    ax.text(0.5,1.2, "Mixed environment", transform=ax.transAxes, ha="center", fontsize=20)#, fontweight="bold")

    ax.legend(loc="lower left", bbox_to_anchor=(-0.1, -0.1), fontsize=14)
    tight_layout(w_pad=-8.0, h_pad=2.0)
    
    PyPlot.savefig(out_filename, dpi=300)
    PyPlot.close()
end
