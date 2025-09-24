import 'package:flutter/material.dart';

import '../../theme/app_colors.dart';

class LinkedAccountsPage extends StatefulWidget {
  const LinkedAccountsPage({super.key});

  static const route = '/linked-accounts';

  @override
  State<LinkedAccountsPage> createState() => _LinkedAccountsPageState();
}

class _LinkedAccountsPageState extends State<LinkedAccountsPage> {
  bool googleLinked = true;
  bool appleLinked = false;
  bool slackLinked = true;

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: SafeArea(
        child: ListView(
          padding: const EdgeInsets.fromLTRB(24, 16, 24, 32),
          children: [
            _header(context, 'Linked Accounts'),
            const SizedBox(height: 24),
            _linkedTile(
              icon: Icons.g_translate,
              title: 'Google',
              subtitle: 'Connected as olivia.wilson@gmail.com',
              value: googleLinked,
              onChanged: (value) => setState(() => googleLinked = value),
            ),
            _linkedTile(
              icon: Icons.apple,
              title: 'Apple',
              subtitle: 'Connect to sync with Apple ID',
              value: appleLinked,
              onChanged: (value) => setState(() => appleLinked = value),
            ),
            _linkedTile(
              icon: Icons.chat_bubble_outline,
              title: 'Slack',
              subtitle: 'Receive updates in Slack',
              value: slackLinked,
              onChanged: (value) => setState(() => slackLinked = value),
            ),
          ],
        ),
      ),
    );
  }

  Widget _linkedTile({
    required IconData icon,
    required String title,
    required String subtitle,
    required bool value,
    required ValueChanged<bool> onChanged,
  }) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 12),
      child: Container(
        padding: const EdgeInsets.all(20),
        decoration: BoxDecoration(
          color: Colors.white,
          borderRadius: BorderRadius.circular(20),
        ),
        child: Row(
          children: [
            Container(
              width: 48,
              height: 48,
              decoration: BoxDecoration(
                color: AppColors.primaryBlue.withOpacity(0.12),
                borderRadius: BorderRadius.circular(16),
              ),
              child: Icon(icon, color: AppColors.primaryBlue),
            ),
            const SizedBox(width: 16),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    title,
                    style: const TextStyle(
                      fontWeight: FontWeight.w700,
                      fontSize: 16,
                    ),
                  ),
                  const SizedBox(height: 4),
                  Text(
                    subtitle,
                    style: const TextStyle(color: AppColors.textSecondary),
                  ),
                ],
              ),
            ),
            Switch.adaptive(
              value: value,
              onChanged: onChanged,
              activeColor: AppColors.primaryBlue,
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
