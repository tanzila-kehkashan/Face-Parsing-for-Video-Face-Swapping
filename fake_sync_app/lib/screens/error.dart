import 'package:flutter/material.dart';

import '../services/navigation_service.dart';

class ErrorSampleScreen extends StatelessWidget {
  const ErrorSampleScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        leading: IconButton(
          icon: const Icon(Icons.arrow_back, color: Colors.white),
          onPressed: () => NavigationService.pushNamed('/login'),
        ),
        title: const Text('FAKESYNC STUDIO'),
        flexibleSpace: Container(
          decoration: const BoxDecoration(
            gradient: LinearGradient(
              colors: [Color(0xFF7E57C2), Color(0xFF5E35B1)],
            ),
          ),
        ),
      ),
      body: Container(
        decoration: const BoxDecoration(
          gradient: LinearGradient(
            begin: Alignment.topLeft,
            end: Alignment.bottomRight,
            colors: [Color(0xFF1E1E1E), Color(0xFF2D2D2D)],
          ),
        ),
        child: Padding(
          padding: const EdgeInsets.all(24.0),
          child: Center(
            child: SingleChildScrollView(
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  // Animated error icon
                  RotationTransition(
                    turns: const AlwaysStoppedAnimation(-0.1),
                    child: Container(
                      padding: const EdgeInsets.all(20),
                      decoration: BoxDecoration(
                        shape: BoxShape.circle,
                        gradient: RadialGradient(
                          colors: [
                            Colors.red.shade700.withOpacity(0.3),
                            Colors.transparent,
                          ],
                        ),
                      ),
                      child: const Icon(
                        Icons.warning_amber_rounded,
                        size: 80,
                        color: Colors.red,
                      ),
                    ),
                  ),
                  const SizedBox(height: 30),

                  // Error title
                  const Text(
                    'Content Processing Error',
                    style: TextStyle(
                      fontSize: 28,
                      fontWeight: FontWeight.bold,
                      color: Colors.white,
                    ),
                    textAlign: TextAlign.center,
                  ),
                  const SizedBox(height: 20),

                  // Error description
                  Text(
                    'We encountered an issue while processing your content. '
                        'Please ensure you\'ve uploaded valid face images and videos.',
                    style: TextStyle(
                      fontSize: 16,
                      color: Colors.grey.shade400,
                      height: 1.5,
                    ),
                    textAlign: TextAlign.center,
                  ),
                  const SizedBox(height: 30),

                  // Error details box
                  Container(
                    padding: const EdgeInsets.all(20),
                    decoration: BoxDecoration(
                      color: Colors.red.withOpacity(0.1),
                      borderRadius: BorderRadius.circular(16),
                      border: Border.all(
                        color: Colors.red.withOpacity(0.3),
                        width: 1.5,
                      ),
                    ),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Row(
                          children: [
                            Icon(
                              Icons.error_outline,
                              color: Colors.red.shade300,
                              size: 20,
                            ),
                            const SizedBox(width: 10),
                            Text(
                              'Error Details',
                              style: TextStyle(
                                fontSize: 18,
                                fontWeight: FontWeight.w600,
                                color: Colors.red.shade300,
                              ),
                            ),
                          ],
                        ),
                        const SizedBox(height: 15),
                        Text(
                          '• No detectable faces in source image\n',
                          style: TextStyle(
                            fontSize: 15,
                            color: Colors.grey.shade300,
                            height: 1.6,
                          ),
                        ),
                      ],
                    ),
                  ),
                  const SizedBox(height: 30),

                  // Solutions section
                  Container(
                    padding: const EdgeInsets.all(20),
                    decoration: BoxDecoration(
                      color: Colors.purple.withOpacity(0.1),
                      borderRadius: BorderRadius.circular(16),
                      border: Border.all(
                        color: Colors.purple.withOpacity(0.3),
                        width: 1.5,
                      ),
                    ),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Row(
                          children: [
                            Icon(
                              Icons.lightbulb_outline,
                              color: Colors.purple.shade300,
                              size: 20,
                            ),
                            const SizedBox(width: 10),
                            Text(
                              'Recommended Solutions',
                              style: TextStyle(
                                fontSize: 18,
                                fontWeight: FontWeight.w600,
                                color: Colors.purple.shade300,
                              ),
                            ),
                          ],
                        ),
                        const SizedBox(height: 15),
                        _buildSolutionItem('Use clear, front-facing portrait photos'),
                        _buildSolutionItem('Ensure good lighting and minimal shadows'),
                        _buildSolutionItem('Avoid group photos or multiple faces'),
                        _buildSolutionItem('Use high-resolution images (min 512px)'),
                        _buildSolutionItem('Check file formats (JPG, PNG, MP4)'),
                      ],
                    ),
                  ),
                  const SizedBox(height: 40),

                  // Action buttons
                  Row(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      _buildActionButton(
                        'Try Again',
                        Icons.refresh,
                        Colors.purple,
                            () {},
                      ),
                      const SizedBox(width: 20),
                      _buildActionButton(
                        'Go Back',
                        Icons.arrow_back,
                        const Color(0xFF2D2D2D),
                            () {},
                      ),
                    ],
                  ),
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildSolutionItem(String text) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 6),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Icon(
            Icons.check_circle_outline,
            color: Colors.green,
            size: 18,
          ),
          const SizedBox(width: 10),
          Expanded(
            child: Text(
              text,
              style: TextStyle(
                fontSize: 15,
                color: Colors.grey.shade300,
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildActionButton(String text, IconData icon, Color color, VoidCallback onPressed) {
    return ElevatedButton.icon(
      onPressed: onPressed,
      icon: Icon(icon, size: 20),
      label: Text(text),
      style: ElevatedButton.styleFrom(
        foregroundColor: Colors.white,
        backgroundColor: color,
        padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 14),
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(30),
          side: BorderSide(
            color: color == const Color(0xFF2D2D2D)
                ? Colors.purple.shade700
                : Colors.transparent,
          ),
        ),
        elevation: 3,
      ),
    );
  }
}