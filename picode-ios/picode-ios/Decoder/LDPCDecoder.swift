import Foundation

/// Soft-decision LDPC decoder matching `picode.ecc.ldpc.LDPC` (pyldpc log-domain belief
/// propagation). The code structure is exported from Python by
/// `picode-model/scripts/export_ldpc.py` into `PicodeLDPC.json`.
struct LDPCDecoder {
    struct Spec: Decodable {
        let n: Int
        let k: Int
        let seed: Int
        let snrs: [Double]
        /// Variable (codeword bit) indices taking part in each parity check.
        let checks: [[Int]]
        /// Codeword position of each message bit (systematic code).
        let messagePositions: [Int]
    }

    struct Result {
        let payload: [Int]
        let success: Bool
        let snr: Double?
    }

    let spec: Spec
    let maxIterations: Int

    init(spec: Spec, maxIterations: Int = 200) {
        self.spec = spec
        self.maxIterations = maxIterations
    }

    /// Load the spec bundled with the app (or a test bundle).
    static func bundled(in bundle: Bundle = .main, name: String = "PicodeLDPC") -> LDPCDecoder? {
        guard let url = bundle.url(forResource: name, withExtension: "json"),
              let data = try? Data(contentsOf: url),
              let spec = try? JSONDecoder().decode(Spec.self, from: data) else {
            return nil
        }
        return LDPCDecoder(spec: spec)
    }

    /// Decode decoder logits (P(bit=1) = sigmoid(logit)), retrying lower SNRs until the
    /// parity check passes — lower SNR trusts confident-but-wrong bits less.
    func decode(logits: [Float]) -> Result {
        precondition(logits.count >= spec.n, "expected \(spec.n) logits")
        var last: [Int] = []
        for snr in spec.snrs {
            let (codeword, ok) = beliefPropagation(logits: Array(logits.prefix(spec.n)), snr: snr)
            last = spec.messagePositions.map { codeword[$0] }
            if ok {
                return Result(payload: last, success: true, snr: snr)
            }
        }
        return Result(payload: last, success: false, snr: nil)
    }

    /// Log-domain sum-product decoding. Returns the hard codeword and parity status.
    func beliefPropagation(logits: [Float], snr: Double) -> ([Int], Bool) {
        let n = spec.n
        let variance = pow(10.0, -snr / 10.0)
        // pyldpc maps P(bit=1)=p to y = 1 - 2p (+1 means bit 0) and uses Lc = 2y / var.
        let channel: [Double] = logits.map { logit in
            let p = 1.0 / (1.0 + exp(-Double(logit)))
            return 2.0 * (1.0 - 2.0 * p) / variance
        }
        // Messages indexed [check][slot], slot = position of the variable within the check.
        var q = spec.checks.map { $0.map { channel[$0] } }
        var r = spec.checks.map { [Double](repeating: 0, count: $0.count) }
        var codeword = channel.map { $0 <= 0 ? 1 : 0 }
        if satisfiesParity(codeword) {
            return (codeword, true)
        }
        for _ in 0..<maxIterations {
            // Check-node update: r = 2 atanh(prod_{others} tanh(q / 2))
            for (c, vars) in spec.checks.enumerated() {
                let t = q[c].map { tanh($0 / 2.0) }
                for slot in vars.indices {
                    var prod = 1.0
                    for (other, value) in t.enumerated() where other != slot {
                        prod *= value
                    }
                    prod = min(max(prod, -0.999_999_999_999), 0.999_999_999_999)
                    r[c][slot] = 2.0 * atanh(prod)
                }
            }
            // Variable-node update and posterior.
            var posterior = channel
            for (c, vars) in spec.checks.enumerated() {
                for (slot, v) in vars.enumerated() {
                    posterior[v] += r[c][slot]
                }
            }
            for (c, vars) in spec.checks.enumerated() {
                for (slot, v) in vars.enumerated() {
                    q[c][slot] = posterior[v] - r[c][slot]
                }
            }
            codeword = posterior.map { $0 <= 0 ? 1 : 0 }
            if satisfiesParity(codeword) {
                return (codeword, true)
            }
        }
        return (codeword, false)
    }

    private func satisfiesParity(_ codeword: [Int]) -> Bool {
        spec.checks.allSatisfy { vars in vars.reduce(0) { $0 ^ codeword[$1] } == 0 }
    }
}
