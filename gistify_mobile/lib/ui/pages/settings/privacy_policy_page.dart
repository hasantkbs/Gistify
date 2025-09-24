import 'package:flutter/material.dart';

import '../../theme/app_colors.dart';

class PrivacyPolicyPage extends StatelessWidget {
  const PrivacyPolicyPage({super.key});

  static const route = '/privacy-policy';

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: SafeArea(
        child: ListView(
          padding: const EdgeInsets.fromLTRB(24, 16, 24, 32),
          children: [
            _header(context, 'Privacy Policy'),
            const SizedBox(height: 24),
            const Text(
              'Your privacy is important to us. This policy explains what data we collect, how we use it, and your rights.',
              style: TextStyle(color: AppColors.textSecondary),
            ),
            const SizedBox(height: 16),
            const _BodyParagraph(
              title: 'Information We Collect',
              body:
                  'We collect information you provide directly, such as account details, uploaded files, and support requests.',
            ),
            const _BodyParagraph(
              title: 'How We Use Information',
              body:
                  'Data is used to provide, maintain, and improve our services, communicate with you, and ensure account security.',
            ),
            const _BodyParagraph(
              title: 'Your Choices',
              body:
                  'You can update your data in account settings, export your summaries, or request deletion from our help center.',
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
