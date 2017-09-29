function bleu{T<:Array{String,1}}(hyps::T, refs::Array{T}; captioning=false)
    if length(hyps) != length(refs)
        error("Hypothesis/references dimension mismatch")
    end

    N = 4
    hypothesis_length = references_length = 0
    clipped = Array{Float64}(N)
    total = Array{Float64}(N)

    for (hyp,ref) in zip(hyps, refs)
        # word or n-gram counts
        hypwc = map(n -> ngram_counts(hyp,n), 1:N)
        refwc = map(r -> map(n -> ngram_counts(r,n), 1:N), ref)

        hyplen = sum(values(hypwc[1]))
        reflen = map(x->sum(values(x[1])), refwc)
        diffs = sort(map(x -> (abs(x - hyplen), x), reflen))

        hypothesis_length += hyplen
        references_length += diffs[1][2]

        for n = 1:N
            c1 = collect(values(hypwc[n]))
            c2 = map(
                x -> mapreduce(r -> get(r[n], x, 0), max, refwc),
                keys(hypwc[n]))
            if length(c1) >= 1
                clipped[n] += sum(map(k->min(c1[k],c2[k]), 1:length(c1)))
                total[n] += sum(c1)
            end
        end
    end

    scores = zeros(N)
    if length(hyps) != 0
        scores = map(i -> clipped[i]/total[i], 1:N)
    end

    # brevity penalty
    bp = 1
    if hypothesis_length < references_length
        bp = exp(1-references_length/hypothesis_length)
    end

    bleuN = map(
        k -> bp * exp(mean(map(log, scores[(captioning ? 1 : k):k]))), 1:N)

    (bleuN, bp, hypothesis_length, references_length)
end

report_bleu(x::Tuple{Array{Float64,1},Int64,Int64,Int64}) = report_bleu(x...)
function report_bleu(
    bleuN::Array{Float64,1}, bp::Int64, hyplen::Int64, reflen::Int64)
    @printf("\nBLEU = %.1f/%.1f/%.1f/%.1f ", map(i->i*100,scores)...)
end
