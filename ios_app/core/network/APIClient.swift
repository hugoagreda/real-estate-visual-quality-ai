import Foundation
import UIKit

class APIClient {
    private let baseURL = "https://mammiferous-brenda-unjumbled.ngrok-free.dev"

    // =====================================================
    // SCORE IMAGE
    // =====================================================

    func scoreImage(_ image: UIImage) async throws -> ScoreResponse {

        guard let url = URL(string: "\(baseURL)/score") else {
            throw URLError(.badURL)
        }

        var request = URLRequest(url: url)
        request.httpMethod = "POST"

        let boundary = UUID().uuidString
        request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")

        guard let imageData = image.jpegData(compressionQuality: 0.9) else {
            throw URLError(.cannotEncodeContentData)
        }

        request.httpBody = createMultipartBody(
            data: imageData,
            boundary: boundary,
            fileName: "image.jpg"
        )

        let (data, response) = try await URLSession.shared.data(for: request)

        guard let httpResponse = response as? HTTPURLResponse else {
            throw URLError(.badServerResponse)
        }

        if httpResponse.statusCode != 200 {
            let text = String(data: data, encoding: .utf8) ?? ""
            print("❌ Backend error:", text)
            throw URLError(.badServerResponse)
        }

        let decoded = try JSONDecoder().decode(ScoreResponse.self, from: data)
        return decoded
    }

    // =====================================================
    // MULTIPART BODY
    // =====================================================

    private func createMultipartBody(
        data: Data,
        boundary: String,
        fileName: String
    ) -> Data {

        var body = Data()

        let lineBreak = "\r\n"

        body.append("--\(boundary)\(lineBreak)")
        body.append("Content-Disposition: form-data; name=\"file\"; filename=\"\(fileName)\"\(lineBreak)")
        body.append("Content-Type: image/jpeg\(lineBreak)\(lineBreak)")
        body.append(data)
        body.append(lineBreak)
        body.append("--\(boundary)--\(lineBreak)")

        return body
    }
}

// =====================================================
// RESPONSE MODEL
// =====================================================

struct ScoreResponse: Codable {
    let score: Float
    let margin: Float
}

// =====================================================
// DATA EXTENSION
// =====================================================

extension Data {
    mutating func append(_ string: String) {
        if let data = string.data(using: .utf8) {
            append(data)
        }
    }
}
