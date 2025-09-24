import 'package:flutter/material.dart';

import '../../theme/app_colors.dart';

class TermsPage extends StatelessWidget {
  const TermsPage({super.key});

  static const route = '/terms';

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: SafeArea(
        child: ListView(
          padding: const EdgeInsets.fromLTRB(24, 16, 24, 32),
          children: [
            _header(context, 'Terms of Service'),
            const SizedBox(height: 24),
            const Text(
              'Last updated: September 24, 2024',
              style: TextStyle(color: AppColors.textSecondary),
            ),
            const SizedBox(height: 16),
            const _BodyParagraph(
              title: '1. Acceptance of Terms',
              body:
                  'By accessing Gistify you agree to comply with these Terms of Service and all applicable laws and regulations.',
            ),
            const _BodyParagraph(
              title: '2. Use License',
              body:
                  'Permission is granted to temporarily download one copy of the materials for personal, non-commercial transitory viewing only.',
            ),
            const _BodyParagraph(
              title: '3. User Responsibilities',
              body:
                  'You are responsible for maintaining the confidentiality of your account and for all activities that occur under your account.',
            ),
          ],
        ),
      ),
    );
  }

  Widget _header(BuildContext context, String title) {
    return Row(
      children: [
        IconButton(
          onPressed: () => Navigator.pop(context),
          icon: const Icon(Icons.arrow_back_ios_new),
        ),
        const Spacer(),
        Text(
          title,
          style: const TextStyle(
            fontSize: 20,
            fontWeight: FontWeight.w700,
            color: AppColors.textPrimary,
          ),
        ),
        const Spacer(),
        const SizedBox(width: 48),
      ],
    );
  }
}

class _BodyParagraph extends StatelessWidget {
  const _BodyParagraph({required this.title, required this.body});

  final String title;
  final String body;

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 20),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            title,
            style: const TextStyle(fontWeight: FontWeight.w700, fontSize: 16),
          ),
          const SizedBox(height: 8),
          Text(body, style: const TextStyle(color: AppColors.textSecondary)),
        ],
      ),
    );
  }
}
