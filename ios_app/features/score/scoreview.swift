import SwiftUI

struct ScoreView: View {

    @State private var selectedImage: UIImage?
    @State private var showingPicker = false
    @State private var scoreText = "No image"
    @State private var loading = false

    let api = APIClient()

    var body: some View {

        VStack(spacing: 20) {

            if let img = selectedImage {
                Image(uiImage: img)
                    .resizable()
                    .scaledToFit()
                    .frame(height: 250)
            }

            Text(scoreText)
                .font(.title2)

            if loading {
                ProgressView()
            }

            Button("Seleccionar imagen") {
                showingPicker = true
            }

            Button("Score 🔥") {
                Task {
                    await sendToBackend()
                }
            }
            .disabled(selectedImage == nil)

        }
        .sheet(isPresented: $showingPicker) {
            ImagePicker(selectedImage: $selectedImage)
        }
        .padding()
    }

    // =====================================================
    // SEND IMAGE TO BACKEND
    // =====================================================

    func sendToBackend() async {

        guard let image = selectedImage else { return }

        loading = true
        scoreText = "Scoring..."

        do {

            let result = try await api.scoreImage(image)

            scoreText = String(format: "🔥 Score %.3f", result.score)

        } catch {
            scoreText = "❌ Error"
            print(error)
        }

        loading = false
    }
}
